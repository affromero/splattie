import type { AutoBlinkConfig } from '../types';

export class AutoBlink {
  private nextBlinkTime: number;
  private blinkValue = 0;
  private config: AutoBlinkConfig;

  constructor(config: AutoBlinkConfig = { interval: [2000, 7000], duration: 150 }) {
    this.config = config;
    this.nextBlinkTime = this.scheduleNext();
  }

  private scheduleNext(): number {
    const [min, max] = this.config.interval;
    return performance.now() + min + Math.random() * (max - min);
  }

  getWeights(): Record<string, number> {
    const now = performance.now();

    if (now > this.nextBlinkTime) {
      const elapsed = now - this.nextBlinkTime;
      if (elapsed < this.config.duration) {
        this.blinkValue = Math.sin((elapsed / this.config.duration) * Math.PI);
      } else {
        this.blinkValue = 0;
        this.nextBlinkTime = this.scheduleNext();
      }
    }

    return {
      eyeBlinkLeft: this.blinkValue,
      eyeBlinkRight: this.blinkValue,
    };
  }
}
