import * as THREE from 'three';

export class HitDetector {
  private bgR = 14;
  private bgG = 14;
  private bgB = 20;
  private pixel = new Uint8Array(4);

  setBackgroundColor(color: number): void {
    this.bgR = (color >> 16) & 0xff;
    this.bgG = (color >> 8) & 0xff;
    this.bgB = color & 0xff;
  }

  check(
    renderer: THREE.WebGLRenderer,
    clientX: number,
    clientY: number,
    canvasWidth: number,
    canvasHeight: number,
  ): boolean {
    const gl = renderer.getContext();
    const dpr = renderer.getPixelRatio();
    const px = Math.floor(clientX * dpr);
    const py = Math.floor((canvasHeight - clientY) * dpr);

    if (px < 0 || py < 0) return false;

    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.pixel);

    const diff =
      Math.abs(this.pixel[0] - this.bgR) +
      Math.abs(this.pixel[1] - this.bgG) +
      Math.abs(this.pixel[2] - this.bgB);

    return diff > 40;
  }
}
