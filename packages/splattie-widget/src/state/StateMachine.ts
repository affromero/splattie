import type { StateDefinition, TransitionConfig, WidgetConfig } from '../types';
import { interpolateTransition } from './Interpolator';

const DEFAULT_TRANSITION: TransitionConfig = { duration: 0.3, easing: 'ease-out' };

export class StateMachine {
  private states: Record<string, StateDefinition>;
  private transitions: Record<string, TransitionConfig>;
  private currentName: string;
  private targetName: string | null = null;
  private fromState: StateDefinition;
  private targetState: StateDefinition | null = null;
  private transitionElapsed = 0;
  private transitionConfig: TransitionConfig = DEFAULT_TRANSITION;

  frozen = false;
  currentFrame: StateDefinition;

  constructor(config: WidgetConfig) {
    this.states = config.states;
    this.transitions = config.transitions;
    const firstState = Object.keys(this.states)[0] ?? 'idle';
    this.currentName = firstState;
    this.fromState = this.states[firstState];
    this.currentFrame = { ...this.fromState };
  }

  get activeStateName(): string {
    return this.targetName ?? this.currentName;
  }

  forceState(stateName: string): void {
    if (!this.states[stateName]) return;
    this.currentName = stateName;
    this.targetName = null;
    this.targetState = null;
    this.fromState = this.states[stateName];
    this.currentFrame = { ...this.states[stateName] };
  }

  updateStates(states: Record<string, StateDefinition>): void {
    this.states = states;
    if (this.states[this.currentName]) {
      this.fromState = this.states[this.currentName];
    }
    if (this.targetName && this.states[this.targetName]) {
      this.targetState = this.states[this.targetName];
    }
    if (!this.targetName && this.states[this.currentName]) {
      this.currentFrame = { ...this.states[this.currentName] };
    }
  }

  transitionTo(stateName: string): void {
    if (stateName === this.currentName && !this.targetName) return;
    if (stateName === this.targetName) return;
    if (!this.states[stateName]) return;

    this.fromState = { ...this.currentFrame };
    this.targetName = stateName;
    this.targetState = this.states[stateName];
    this.transitionElapsed = 0;

    const key = `${this.currentName}->${stateName}`;
    const wildcardKey = `*->${stateName}`;
    this.transitionConfig =
      this.transitions[key] ?? this.transitions[wildcardKey] ?? DEFAULT_TRANSITION;
  }

  update(deltaTime: number): void {
    if (this.frozen) return;
    if (!this.targetState || !this.targetName) return;

    this.transitionElapsed += deltaTime;
    const { state, done } = interpolateTransition(
      this.fromState,
      this.targetState,
      this.transitionElapsed,
      this.transitionConfig,
    );

    this.currentFrame = state;

    if (done) {
      this.currentName = this.targetName;
      this.fromState = this.targetState;
      this.targetName = null;
      this.targetState = null;
    }
  }
}
