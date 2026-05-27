/**
 * FLAME expression blendshape basis — loads the binary exported by
 * export_expression_basis.py and applies per-splat position offsets
 * directly to Spark's PackedSplats buffer (half-float encoding).
 */

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

function toHalf(f: number): number {
  _f32[0] = f;
  const i = _i32[0];
  const s = (i >>> 16) & 0x8000;
  const e = ((i >>> 23) & 0xff) - 112;
  const m = i & 0x7fffff;
  if (e <= 0) return s;
  if (e > 30) return s | 0x7c00;
  return s | (e << 10) | (m >>> 13);
}

export interface ExprBasisData {
  numVertices: number;
  numExpressions: number;
  basis: Float32Array;
}

export async function loadExpressionBasis(url: string): Promise<ExprBasisData> {
  const res = await fetch(url);
  const buf = await res.arrayBuffer();
  const header = new DataView(buf, 0, 12);
  const magic = String.fromCharCode(
    header.getUint8(0), header.getUint8(1), header.getUint8(2), header.getUint8(3),
  );
  if (magic !== 'EXPR') throw new Error(`Invalid expression basis magic: ${magic}`);

  const numVertices = header.getUint32(4, true);
  const numExpressions = header.getUint32(8, true);
  const basis = new Float32Array(buf, 12);

  return { numVertices, numExpressions, basis };
}

export class ExpressionBasisApplier {
  private basis: ExprBasisData;
  private baselinePositions: Float32Array;
  private numSplats: number;
  private prevWeights: Float32Array;
  private splatMask: Float32Array;

  constructor(basis: ExprBasisData, baselinePositions: Float32Array, jawY?: number) {
    this.basis = basis;
    this.baselinePositions = baselinePositions;
    this.numSplats = Math.min(baselinePositions.length / 3, basis.numVertices);
    this.prevWeights = new Float32Array(basis.numExpressions);

    this.splatMask = new Float32Array(this.numSplats);
    const jy = jawY ?? -0.016;
    for (let i = 0; i < this.numSplats; i++) {
      const y = baselinePositions[i * 3 + 1];
      const z = baselinePositions[i * 3 + 2];
      // Full influence above jaw; fade out below jaw (beard/chin area)
      // Also reduce for splats far behind the face (z very negative = back of head)
      const yFade = y > jy ? 1.0 : Math.max(0, 1.0 - (jy - y) / 0.025);
      const zFade = z > -0.06 ? 1.0 : Math.max(0, 1.0 - (-0.06 - z) / 0.02);
      this.splatMask[i] = yFade * zFade;
    }
  }

  apply(
    packedArray: Uint32Array,
    weights: Record<string, number>,
    scale: number = 3.0,
  ): boolean {
    const { basis, numExpressions } = this.basis;
    const w = new Float32Array(numExpressions);
    for (let j = 0; j < numExpressions; j++) {
      w[j] = (weights[`expr_${j}`] ?? 0) * scale;
    }

    let changed = false;
    for (let j = 0; j < numExpressions; j++) {
      if (Math.abs(w[j] - this.prevWeights[j]) > 0.001) { changed = true; break; }
    }
    if (!changed) return false;
    this.prevWeights.set(w);

    const bl = this.baselinePositions;
    const mask = this.splatMask;
    for (let i = 0; i < this.numSplats; i++) {
      const m = mask[i];
      if (m < 0.001) {
        continue;
      }
      let dx = 0, dy = 0, dz = 0;
      const bOff = i * numExpressions * 3;
      for (let j = 0; j < numExpressions; j++) {
        if (w[j] === 0) continue;
        const off = bOff + j * 3;
        dx += w[j] * basis[off];
        dy += w[j] * basis[off + 1];
        dz += w[j] * basis[off + 2];
      }

      const nx = bl[i * 3] + dx * m;
      const ny = bl[i * 3 + 1] + dy * m;
      const nz = bl[i * 3 + 2] + dz * m;

      // Packed format: [i4+1] = halfX | halfY<<16, [i4+2] = halfZ | (keep upper 16 bits)
      const i4 = i * 4;
      packedArray[i4 + 1] = toHalf(nx) | (toHalf(ny) << 16);
      packedArray[i4 + 2] = (packedArray[i4 + 2] & 0xFFFF0000) | toHalf(nz);
    }

    return true;
  }
}
