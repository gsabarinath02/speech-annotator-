export function buildRawAudioConstraints(): MediaStreamConstraints {
  return {
    audio: {
      channelCount: { ideal: 1 },
      sampleRate: { ideal: 48_000 },
      sampleSize: { ideal: 32 },
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  };
}

