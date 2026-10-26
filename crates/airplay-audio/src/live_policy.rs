//! Local live-audio buffering. These limits do not configure receiver latency.
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct LiveStreamOptions {
    pub buffer_ms: u32,
    pub startup_ms: u32,
    pub refill_ms: u32,
    pub resume_ms: u32,
    pub startup_timeout: Duration,
    pub sender_capacity: usize,
    pub max_age: Option<Duration>,
}

impl Default for LiveStreamOptions {
    fn default() -> Self {
        Self {
            buffer_ms: 2000,
            startup_ms: 1000,
            refill_ms: 800,
            resume_ms: 200,
            startup_timeout: Duration::from_secs(5),
            sender_capacity: 8,
            max_age: None,
        }
    }
}

impl LiveStreamOptions {
    pub fn low_latency() -> Self {
        Self {
            buffer_ms: 80,
            startup_ms: 24,
            refill_ms: 24,
            resume_ms: 8,
            startup_timeout: Duration::from_millis(500),
            sender_capacity: 2,
            max_age: Some(Duration::from_millis(120)),
        }
    }

    pub(crate) fn validate(self) -> airplay_core::error::Result<()> {
        if self.buffer_ms == 0
            || self.startup_ms > self.buffer_ms
            || self.refill_ms > self.buffer_ms
            || self.resume_ms > self.buffer_ms
            || self.sender_capacity == 0
            || self.startup_timeout.is_zero()
        {
            return Err(airplay_core::error::StreamingError::Encoding(
                "invalid live buffer policy".into(),
            )
            .into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct LiveDiagnosticsSnapshot {
    pub queued_blocks: usize,
    pub queue_drops: u64,
    pub stale_drops: u64,
    pub packets_sent: u64,
    pub send_errors: u64,
    pub underruns: u64,
    pub encoder_buffer_ms: u64,
    pub max_queue_age_us: u64,
    pub capture_to_send_p95_us: Option<u64>,
    pub first_packet_age_us: Option<u64>,
}

/// Bounded measurements shared by capture input, encoder and UDP sender.
#[derive(Default)]
pub struct LiveDiagnostics {
    pub(crate) queue_drops: AtomicU64,
    pub(crate) evicted_samples: AtomicU64,
    pub(crate) stale_drops: AtomicU64,
    pub(crate) packets_sent: AtomicU64,
    pub(crate) send_errors: AtomicU64,
    pub(crate) underruns: AtomicU64,
    pub(crate) encoder_buffer_ms: AtomicU64,
    pub(crate) max_queue_age_us: AtomicU64,
    pub(crate) force_sync: AtomicBool,
    ages: Mutex<VecDeque<u64>>,
    first_packet_age_us: Mutex<Option<u64>>,
}

impl LiveDiagnostics {
    pub(crate) fn observe_queue_age(&self, captured: Instant) {
        self.max_queue_age_us
            .fetch_max(captured.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    pub(crate) fn observe_send(&self, captured: Option<Instant>) {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
        if let Some(captured) = captured {
            let age = captured.elapsed().as_micros() as u64;
            let mut first = self.first_packet_age_us.lock().unwrap();
            first.get_or_insert(age);
            drop(first);
            let mut ages = self.ages.lock().unwrap();
            if ages.len() == 256 {
                ages.pop_front();
            }
            ages.push_back(age);
        }
    }

    pub fn snapshot(&self, queued_blocks: usize) -> LiveDiagnosticsSnapshot {
        let mut ages: Vec<_> = self.ages.lock().unwrap().iter().copied().collect();
        ages.sort_unstable();
        LiveDiagnosticsSnapshot {
            queued_blocks,
            queue_drops: self.queue_drops.load(Ordering::Relaxed),
            stale_drops: self.stale_drops.load(Ordering::Relaxed),
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            send_errors: self.send_errors.load(Ordering::Relaxed),
            underruns: self.underruns.load(Ordering::Relaxed),
            encoder_buffer_ms: self.encoder_buffer_ms.load(Ordering::Relaxed),
            max_queue_age_us: self.max_queue_age_us.load(Ordering::Relaxed),
            capture_to_send_p95_us: ages.get(ages.len() * 95 / 100).copied(),
            first_packet_age_us: *self.first_packet_age_us.lock().unwrap(),
        }
    }
}

pub(crate) fn expired(captured: Option<Instant>, max_age: Option<Duration>) -> bool {
    captured
        .zip(max_age)
        .is_some_and(|(at, limit)| at.elapsed() > limit)
}

pub(crate) type SharedDiagnostics = Arc<LiveDiagnostics>;

/// Preserve the live RTP/wall-time relationship after a capture underrun.
/// A bounded-age live stream cannot recover missing samples by playing late.
#[derive(Default)]
pub(crate) struct LiveTimeline {
    anchor: Option<(Instant, u64)>,
    offset: u64,
}
impl LiveTimeline {
    pub(crate) fn timestamp(&mut self, source: u64, sample_rate: u32, now: Instant) -> u32 {
        let base = source.saturating_add(self.offset);
        let (at, timestamp) = *self.anchor.get_or_insert((now, base));
        let elapsed = (now.saturating_duration_since(at).as_nanos() * sample_rate as u128 / 1_000_000_000) as u64;
        let lag = timestamp.saturating_add(elapsed).saturating_sub(base);
        if lag > sample_rate as u64 / 50 {
            self.offset = self.offset.saturating_add(lag);
        }
        source.saturating_add(self.offset) as u32
    }
}
#[cfg(test)]
mod timeline_tests {
    use super::*;
    #[test]
    fn resumes_after_capture_gap_without_replaying_lost_time() {
        let now = Instant::now();
        let mut timeline = LiveTimeline::default();
        assert_eq!(timeline.timestamp(0, 44100, now), 0);
        assert_eq!(timeline.timestamp(441, 44100, now + Duration::from_millis(10)), 441);
        assert_eq!(timeline.timestamp(882, 44100, now + Duration::from_millis(120)), 5292);
        assert_eq!(timeline.timestamp(1323, 44100, now + Duration::from_millis(130)), 5733);
    }
    #[test]
    fn tolerates_scheduler_jitter_and_does_not_move_audio_backwards() {
        let now = Instant::now();
        let mut timeline = LiveTimeline::default();
        timeline.timestamp(0, 44100, now);
        assert_eq!(timeline.timestamp(441, 44100, now + Duration::from_millis(25)), 441);
        assert_eq!(timeline.timestamp(4410, 44100, now + Duration::from_millis(30)), 4410);
    }
    #[test]
    fn wraps_rtp_and_reset_starts_a_new_anchor() {
        let now = Instant::now();
        let mut timeline = LiveTimeline::default();
        assert_eq!(timeline.timestamp(u32::MAX as u64 + 10, 44100, now), 9);
        timeline = LiveTimeline::default();
        assert_eq!(timeline.timestamp(100, 44100, now + Duration::from_secs(20)), 100);
    }
}
