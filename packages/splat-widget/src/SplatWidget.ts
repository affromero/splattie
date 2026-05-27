import * as THREE from 'three';
import { SplatEdit, SplatEditSdf, SplatEditSdfType } from '@sparkjsdev/spark';
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
  _stateMachine: StateMachine | null = null;
  private get stateMachine() { return this._stateMachine; }
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
  private blinkEdit: { left: SplatEditSdf; right: SplatEditSdf; edit: SplatEdit } | null = null;

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
      this._stateMachine = new StateMachine(this.config);

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
      this.setupBlinkEdits();
      this.dispatchEvent(new CustomEvent('splatload', { bubbles: true }));

      this.startRenderLoop();
    } catch (err) {
      console.error('splat-widget init failed:', err);
    }
  }

  disconnectedCallback(): void {
    this.spark?.renderer.dispose();
    this.spark = null;
    this.cursor.detach();
  }

  setState(name: string): void {
    this.stateMachine?.transitionTo(name);
  }

  private setupBlinkEdits(): void {
    if (!this.spark || this.spark.bones.length < 5) return;
    const mesh = this.spark.splatMesh as unknown as THREE.Object3D & { edits: SplatEdit[] | null };

    const leftEyePos = this.spark.bones[3].pos;
    const rightEyePos = this.spark.bones[4].pos;

    const leftSdf = new SplatEditSdf({
      type: SplatEditSdfType.SPHERE,
      radius: 0.015,
    });
    leftSdf.position.set(leftEyePos[0], leftEyePos[1], leftEyePos[2]);

    const rightSdf = new SplatEditSdf({
      type: SplatEditSdfType.SPHERE,
      radius: 0.015,
    });
    rightSdf.position.set(rightEyePos[0], rightEyePos[1], rightEyePos[2]);

    const edit = new SplatEdit({
      sdfs: [leftSdf, rightSdf],
      softEdge: 0.005,
    });
    mesh.add(edit);
    if (!mesh.edits) mesh.edits = [];
    mesh.edits.push(edit);
    this.blinkEdit = { left: leftSdf, right: rightSdf, edit };
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

      // Blink via SplatEdit — fade eye splats to transparent
      if (this.blinkEdit) {
        const blink = this.autoBlink.getWeights();
        const blinkVal = blink.eyeBlinkLeft ?? 0;
        // opacity = 1 means fully visible (no blink), 0 means invisible (full blink)
        this.blinkEdit.left.opacity = 1 - blinkVal;
        this.blinkEdit.right.opacity = 1 - blinkVal;
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

    // Eyes (bones 3, 4) — cursor tracking, clamp NDC to ±1 then scale
    const clampedX = Math.max(-1, Math.min(1, this.cursor.ndcX));
    const clampedY = Math.max(-1, Math.min(1, this.cursor.ndcY));
    const eyeYaw = clampedX * 0.09 * tracking.eyes;
    const eyePitch = clampedY * 0.04 * tracking.eyes;
    for (const eyeIdx of [3, 4]) {
      if (eyeIdx >= bones.length) continue;
      const q = new THREE.Quaternion();
      q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), eyeYaw));
      q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -eyePitch));
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
