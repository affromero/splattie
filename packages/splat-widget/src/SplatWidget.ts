import * as THREE from 'three';
import { CameraSphere } from './dimensions/CameraSphere';
import { CursorTracking } from './dimensions/CursorTracking';
import { Expression } from './dimensions/Expression';
import { GhostEffect } from './dimensions/GhostEffect';
import { ObjectRotation } from './dimensions/ObjectRotation';
import { AutoBlink } from './features/AutoBlink';
import { CursorTracker } from './interaction/CursorTracker';
import { SplatEvents } from './interaction/Events';
import { HitDetector } from './interaction/HitDetector';
import { createSparkInstance } from './renderer/SparkSetup';
import type { SparkInstance } from './renderer/SparkSetup';
import { createDefaultConfig, loadConfig } from './state/StateConfig';
import { StateMachine } from './state/StateMachine';
import type { WidgetConfig } from './types';

export class SplatWidget extends HTMLElement {
  private spark: SparkInstance | null = null;
  private stateMachine: StateMachine | null = null;
  private ghost = new GhostEffect();
  private cameraSphere = new CameraSphere();
  private objectRotation = new ObjectRotation();
  private expression = new Expression();
  private cursorTracking = new CursorTracking();
  private autoBlink = new AutoBlink();
  private cursor = new CursorTracker();
  private hitDetector = new HitDetector();
  private events: SplatEvents | null = null;
  private rafId = 0;
  private lastTime = 0;
  private isOnSplat = false;
  private config: WidgetConfig | null = null;

  static get observedAttributes(): string[] {
    return ['src', 'config', 'background', 'width', 'height'];
  }

  async connectedCallback(): Promise<void> {
    try {
      this.style.display = 'block';
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      if (!this.style.width) this.style.width = this.getAttribute('width') ?? '100%';
      if (!this.style.height) this.style.height = this.getAttribute('height') ?? '400px';

      const bgAttr = this.getAttribute('background');
      const bgColor = bgAttr ? parseInt(bgAttr.replace('#', ''), 16) : 0x0e0e14;

      const configUrl = this.getAttribute('config');
      this.config = configUrl ? await loadConfig(configUrl) : createDefaultConfig();
      this.stateMachine = new StateMachine(this.config);

      if (this.config.defaults.autoBlink) {
        this.autoBlink = new AutoBlink(this.config.defaults.autoBlink);
      }

      this.hitDetector.setBackgroundColor(bgColor);
      this.events = new SplatEvents(this);
      this.cursor.attach(this);

      const src = this.getAttribute('src');
      if (!src) { console.warn('splat-widget: no src attribute'); return; }

      console.log('splat-widget: loading', src);
      this.spark = await createSparkInstance(this, src, bgColor);
      console.log('splat-widget: spark ready');
      this.events.attachClick(this);
      this.dispatchEvent(new CustomEvent('splatload', { bubbles: true }));

      this.lastTime = performance.now();
      this.animate();
    } catch (err) {
      console.error('splat-widget init failed:', err);
    }
  }

  disconnectedCallback(): void {
    cancelAnimationFrame(this.rafId);
    this.cursor.detach(this);
    this.spark?.renderer.dispose();
    this.spark = null;
  }

  setState(name: string): void {
    this.stateMachine?.transitionTo(name);
  }

  setExpression(weights: Record<string, number>): void {
    if (!this.stateMachine) return;
    Object.assign(this.stateMachine.currentFrame.expression, weights);
  }

  setCamera(config: { theta?: number; phi?: number; radius?: number }): void {
    if (!this.stateMachine) return;
    Object.assign(this.stateMachine.currentFrame.camera, config);
  }

  private animate = (): void => {
    this.rafId = requestAnimationFrame(this.animate);

    const now = performance.now();
    const deltaTime = (now - this.lastTime) / 1000;
    this.lastTime = now;

    if (!this.spark || !this.stateMachine) return;
    const { renderer, scene, camera, splatMesh } = this.spark;
    const mesh = splatMesh as unknown as THREE.Object3D;

    this.stateMachine.update(deltaTime);
    const frame = this.stateMachine.currentFrame;

    mesh.rotation.set(0, 0, 0);
    mesh.position.set(0, 0, 0);
    this.objectRotation.apply(mesh, frame.rotation);
    this.ghost.apply(mesh, frame.ghost, now / 1000);
    this.cameraSphere.apply(camera, frame.camera);

    renderer.render(scene, camera);

    if (this.cursor.isOnPage) {
      const canvasRect = this.spark.canvas.getBoundingClientRect();
      this.isOnSplat = this.hitDetector.check(
        renderer,
        this.cursor.clientX,
        this.cursor.clientY,
        canvasRect.width,
        canvasRect.height,
      );
    } else {
      this.isOnSplat = false;
    }

    this.events?.update(this.isOnSplat);

    if (this.isOnSplat && this.stateMachine.activeStateName !== 'hover') {
      this.stateMachine.transitionTo('hover');
    } else if (!this.isOnSplat && this.stateMachine.activeStateName === 'hover') {
      this.stateMachine.transitionTo('idle');
    }
  };
}
