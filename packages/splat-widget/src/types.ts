export interface GhostConfig {
  amplitude: number;
  frequency: number;
  wobble: number;
  phase?: number;
}

export interface TrackingConfig {
  eyes: number;
  head: number;
  body?: number;
}

export interface CameraConfig {
  theta: number;
  phi: number;
  radius: number;
  lookAt?: 'auto' | [number, number, number];
  fov?: number;
}

export interface StateDefinition {
  ghost: GhostConfig;
  expression: Record<string, number>;
  camera: CameraConfig;
  rotation: [number, number, number];
  tracking: TrackingConfig;
}

export interface TransitionConfig {
  duration: number;
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'snap';
}

export interface AutoBlinkConfig {
  interval: [number, number];
  duration: number;
}

export interface WidgetConfig {
  version: number;
  defaults: {
    camera: CameraConfig;
    autoBlink?: AutoBlinkConfig;
  };
  states: Record<string, StateDefinition>;
  transitions: Record<string, TransitionConfig>;
}

export interface SplatWidgetEvents {
  splatload: CustomEvent;
  splathover: CustomEvent;
  splatclick: CustomEvent;
  splatleave: CustomEvent;
}
