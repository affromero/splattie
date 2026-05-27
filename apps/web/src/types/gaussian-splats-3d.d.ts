declare module '@mkkellogg/gaussian-splats-3d' {
  export enum SceneRevealMode {
    Default = 0,
    Gradual = 1,
    Instant = 2,
  }

  export enum LogLevel {
    None = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Debug = 4,
  }

  export interface ViewerOptions {
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    rootElement?: HTMLElement;
    sceneRevealMode?: SceneRevealMode;
    logLevel?: LogLevel;
    sharedMemoryForWorkers?: boolean;
    gpuAcceleratedSort?: boolean;
    initialCameraPosition?: [number, number, number];
    initialCameraLookAt?: [number, number, number];
    renderer?: unknown;
    camera?: unknown;
  }

  export interface AddSceneOptions {
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    splatAlphaRemovalThreshold?: number;
    position?: [number, number, number];
    rotation?: [number, number, number, number];
    scale?: [number, number, number];
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(url: string, options?: AddSceneOptions): Promise<void>;
    dispose(): void;
    splatMesh: {
      rotation: { x: number; y: number; z: number };
      position: { x: number; y: number; z: number };
    };
  }

  export class DropInViewer extends Viewer {
    constructor(options?: ViewerOptions);
  }
}
