import { SplatWidget } from './SplatWidget';

export { SplatWidget };
export { StateMachine } from './state/StateMachine';
export { createDefaultConfig, loadConfig, mergeWithDefaults } from './state/StateConfig';
export type {
  AutoBlinkConfig,
  CameraConfig,
  GhostConfig,
  SplatWidgetEvents,
  StateDefinition,
  TrackingConfig,
  TransitionConfig,
  WidgetConfig,
} from './types';

export function register(tagName = 'splattie-widget'): void {
  if (!customElements.get(tagName)) {
    customElements.define(tagName, SplatWidget);
  }
}

register();
