import { describe, expect, it } from 'vitest';
import { StateMachine } from '../src/state/StateMachine';
import { createDefaultConfig } from '../src/state/StateConfig';

describe('StateMachine', () => {
  it('starts with first defined state', () => {
    const sm = new StateMachine(createDefaultConfig());
    expect(sm.activeStateName).toBe('idle');
  });

  it('transitions to a new state', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('hover');
    expect(sm.activeStateName).toBe('hover');
  });

  it('ignores transition to current state', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('idle');
    expect(sm.activeStateName).toBe('idle');
  });

  it('ignores transition to unknown state', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('nonexistent');
    expect(sm.activeStateName).toBe('idle');
  });

  it('interpolates during transition', () => {
    const sm = new StateMachine(createDefaultConfig());
    const idleGhost = sm.currentFrame.ghost.amplitude;

    sm.transitionTo('hover');
    sm.update(0.15);

    expect(sm.currentFrame.ghost.amplitude).not.toBe(idleGhost);
    expect(sm.activeStateName).toBe('hover');
  });

  it('completes transition after full duration', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('hover');
    sm.update(1.0);

    expect(sm.activeStateName).toBe('hover');
    expect(sm.currentFrame.ghost.amplitude).toBeCloseTo(0.008);
  });

  it('snap transition completes immediately', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('click');
    sm.update(0.001);

    expect(sm.activeStateName).toBe('click');
    expect(sm.currentFrame.expression.eyeWideLeft).toBeCloseTo(0.8);
  });

  it('can chain transitions', () => {
    const sm = new StateMachine(createDefaultConfig());
    sm.transitionTo('hover');
    sm.update(1.0);
    sm.transitionTo('click');
    sm.update(1.0);

    expect(sm.activeStateName).toBe('click');
  });
});
