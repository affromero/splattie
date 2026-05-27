import type { CameraConfig, StateDefinition, WidgetConfig } from '../types';

const DEFAULT_CAMERA: CameraConfig = { theta: 0, phi: 75, radius: 2.5, lookAt: 'auto', fov: 50 };

const DEFAULT_STATE: StateDefinition = {
  ghost: { amplitude: 0.005, frequency: 0.5, wobble: 0.3 },
  expression: {},
  camera: DEFAULT_CAMERA,
  rotation: [0, 0, 0],
  tracking: { eyes: 1.0, head: 0.0 },
};

export function createDefaultConfig(): WidgetConfig {
  return {
    version: 1,
    defaults: {
      camera: DEFAULT_CAMERA,
      autoBlink: { interval: [2000, 7000], duration: 150 },
    },
    states: {
      idle: {
        ...DEFAULT_STATE,
        expression: { browDownLeft: 0.12, browOuterUpRight: 0.15, eyeSquintLeft: 0.08 },
      },
      hover: {
        ghost: { amplitude: 0.008, frequency: 0.7, wobble: 0.5 },
        expression: { noseSneerLeft: 0.7, eyeSquintLeft: 0.5, mouthUpperUpLeft: 0.4, browDownLeft: 0.5, browInnerUp: 0.3, eyeWideLeft: 0.4, eyeWideRight: 0.3 },
        camera: { theta: 0, phi: 75, radius: 2.4 },
        rotation: [-3, 0, -2],
        tracking: { eyes: 1.0, head: 0.3 },
      },
      click: {
        ghost: { amplitude: 0.002, frequency: 1.0, wobble: 0.1 },
        expression: { eyeWideLeft: 0.8, eyeWideRight: 0.8, jawOpen: 0.3, browInnerUp: 0.6 },
        camera: { theta: 0, phi: 70, radius: 2.2 },
        rotation: [5, 0, 0],
        tracking: { eyes: 0.5, head: 0.0 },
      },
    },
    transitions: {
      'idle->hover': { duration: 0.3, easing: 'ease-out' },
      'hover->idle': { duration: 0.5, easing: 'ease-in' },
      '*->click': { duration: 0.1, easing: 'snap' },
    },
  };
}

export function mergeWithDefaults(partial: Partial<WidgetConfig>): WidgetConfig {
  const defaults = createDefaultConfig();
  return {
    version: partial.version ?? defaults.version,
    defaults: { ...defaults.defaults, ...partial.defaults },
    states: { ...defaults.states, ...partial.states },
    transitions: { ...defaults.transitions, ...partial.transitions },
  };
}

export async function loadConfig(url: string): Promise<WidgetConfig> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load config: ${res.statusText}`);
  const json = await res.json();
  return mergeWithDefaults(json);
}
