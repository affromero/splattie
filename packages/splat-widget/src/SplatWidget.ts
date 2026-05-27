import * as THREE from 'three';
import { CameraSphere } from './dimensions/CameraSphere';
import { CursorTracking } from './dimensions/CursorTracking';
import { GhostEffect } from './dimensions/GhostEffect';
import { ObjectRotation } from './dimensions/ObjectRotation';
import { AutoBlink } from './features/AutoBlink';
import { CursorTracker } from './interaction/CursorTracker';
import { SplatEvents } from './interaction/Events';
import { HitDetector } from './interaction/HitDetector';
import { createSparkInstance } from './renderer/SparkSetup';
import type { BoneInfo, SparkInstance } from './renderer/SparkSetup';
import { createDefaultConfig, loadConfig } from './state/StateConfig';
import { StateMachine } from './state/StateMachine';
import type { WidgetConfig } from './types';

export class SplatWidget extends HTMLElement {
  private spark: SparkInstance | null = null;
  private stateMachine: StateMachine | null = null;
  private ghost = new GhostEffect();
  private cameraSphere = new CameraSphere();
  private objectRotation = new ObjectRotation();
  private cursorTracking = new CursorTracking();
  private autoBlink = new AutoBlink();
  private cursor = new CursorTracker();
  private hitDetector = new HitDetector();
  private events: SplatEvents | null = null;
  private isOnSplat = false;
  private flyReaction = 0;
  private config: WidgetConfig | null = null;
  private frameCount = 0;

  static get observedAttributes(): string[] {
    return ['src', 'config', 'background', 'width', 'height', 'bones', 'weights'];
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
      if (!src) return;

      const bonesUrl = this.getAttribute('bones') ?? undefined;
      const weightsUrl = this.getAttribute('weights') ?? undefined;

      this.spark = await createSparkInstance(this, src, bgColor, bonesUrl, weightsUrl);
      this.events.attachClick(this);
      this.dispatchEvent(new CustomEvent('splatload', { bubbles: true }));

      this.startRenderLoop();
    } catch (err) {
      console.error('splat-widget init failed:', err);
    }
  }

  disconnectedCallback(): void {
    this.spark?.renderer.dispose();
    this.spark = null;
    this.cursor.detach(this);
  }

  setState(name: string): void {
    this.stateMachine?.transitionTo(name);
  }

  private startRenderLoop(): void {
    const { renderer, scene, camera } = this.spark!;

    renderer.setAnimationLoop(() => {
      this.frameCount++;
      if (!this.stateMachine || !this.spark) return;

      const deltaTime = 1 / 60;
      const now = performance.now() / 1000;
      this.stateMachine.update(deltaTime);
      const frame = this.stateMachine.currentFrame;

      const mesh = this.spark.splatMesh as unknown as THREE.Object3D;

      // Dimension 1: Ghost
      mesh.position.set(0, 0, 0);
      mesh.rotation.set(0, 0, 0);
      this.ghost.apply(mesh, frame.ghost, now);

      // Dimension 4: Object rotation
      this.objectRotation.apply(mesh, frame.rotation);

      // Dimension 3: Camera sphere
      this.cameraSphere.apply(camera, frame.camera);

      // Dimension 2 + 5: Expressions + cursor tracking via SplatSkinning
      if (this.spark.skinning && this.spark.bones.length > 0) {
        this.applySkinning(this.spark.skinning, this.spark.bones, frame);
      }

      // Render
      renderer.render(scene, camera);

      // Hit detection AFTER render
      if (this.frameCount % 3 === 0) {
        if (this.cursor.isOnPage) {
          const rect = this.spark.canvas.getBoundingClientRect();
          this.isOnSplat = this.hitDetector.check(renderer, this.cursor.clientX, this.cursor.clientY, rect.width, rect.height);
        } else {
          this.isOnSplat = false;
        }
        this.events?.update(this.isOnSplat);

        // Auto-transition hover/idle
        if (this.isOnSplat && this.stateMachine.activeStateName !== 'hover') {
          this.stateMachine.transitionTo('hover');
        } else if (!this.isOnSplat && this.stateMachine.activeStateName === 'hover') {
          this.stateMachine.transitionTo('idle');
        }
      }
    });
  }

  private applySkinning(skinning: unknown, bones: BoneInfo[], frame: typeof StateMachine.prototype.currentFrame): void {
    const sk = skinning as {
      setBoneQuatPos: (idx: number, q: THREE.Quaternion, p: THREE.Vector3) => void;
      updateBones: () => void;
    };

    const tracking = frame.tracking;
    const blinkWeights = this.autoBlink.getWeights();

    // Fly reaction smooth in/out
    this.flyReaction += ((this.isOnSplat ? 1 : 0) - this.flyReaction) * 0.1;

    // Eyes (bones 3, 4) — cursor tracking
    const eyeYaw = this.cursor.ndcX * 0.2 * tracking.eyes;
    const eyePitch = this.cursor.ndcY * 0.15 * tracking.eyes;
    for (const eyeIdx of [3, 4]) {
      if (eyeIdx >= bones.length) continue;
      const q = new THREE.Quaternion();
      q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), eyeYaw));
      q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -eyePitch));
      // Blink — close eyes by rotating down
      const blink = blinkWeights.eyeBlinkLeft ?? 0;
      if (blink > 0.01) {
        q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), blink * 0.15));
      }
      sk.setBoneQuatPos(eyeIdx, q, new THREE.Vector3(...bones[eyeIdx].pos));
    }

    // Neck (bone 1) — head follow
    const neckYaw = this.cursor.ndcX * 0.08 * tracking.head;
    const neckPitch = this.cursor.ndcY * 0.05 * tracking.head;
    if (bones.length > 1) {
      const nq = new THREE.Quaternion();
      nq.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), neckYaw));
      nq.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -neckPitch));
      sk.setBoneQuatPos(1, nq, new THREE.Vector3(...bones[1].pos));
    }

    // Jaw (bone 2) — fly reaction
    if (bones.length > 2) {
      const jawAngle = this.flyReaction * 0.15;
      const jq = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), jawAngle);
      sk.setBoneQuatPos(2, jq, new THREE.Vector3(...bones[2].pos));
    }

    sk.updateBones();
  }
}
