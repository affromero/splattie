import { describe, expect, it } from 'vitest';
// Shared validator lives in public/ so the static editor.html can import it at
// runtime via /splattie-format.js; the .d.ts gives us types here.
import { inspectSplattie } from '../../../public/splattie-format.js';

const validManifest = {
  format: 'splattie',
  formatVersion: '0.1.1',
  avatar: { splat: { file: 'head.ply' } },
  widget: { config: 'states.json' },
  animation: { skeleton: { file: 'bone_tree.json' }, weights: { file: 'lbs.json' } },
};
const validFiles = ['head.ply', 'states.json', 'bone_tree.json', 'lbs.json', 'manifest.json'];

describe('inspectSplattie', () => {
  it('accepts a well-formed, version-matched bundle', () => {
    expect(inspectSplattie(validManifest, validFiles, '0.1.1')).toEqual({ ok: true });
  });

  it('rejects a bundle with no manifest as an old-pipeline file', () => {
    const result = inspectSplattie(null, ['andres.ply', 'states.json'], '0.1.1');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('no-manifest');
  });

  it('rejects a file whose manifest is not a splattie', () => {
    const result = inspectSplattie({ format: 'gltf' }, ['scene.gltf'], '0.1.1');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('not-splattie');
  });

  it('reports a version mismatch with both versions in the message', () => {
    const result = inspectSplattie({ ...validManifest, formatVersion: '0.0.9' }, validFiles, '0.1.1');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('version-mismatch');
    expect(result.message).toContain('0.0.9');
    expect(result.message).toContain('0.1.1');
  });

  it('skips the version check when no expected version is available', () => {
    expect(inspectSplattie({ ...validManifest, formatVersion: '0.0.9' }, validFiles, null)).toEqual({ ok: true });
  });

  it('rejects an assetType the widget has no skinning path for', () => {
    const result = inspectSplattie({ ...validManifest, assetType: 'quadruped_mammal' }, validFiles, '0.1.1');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('unsupported-asset-type');
    expect(result.message).toContain('quadruped_mammal');
  });

  it('rejects a bundle missing a file its manifest references', () => {
    const result = inspectSplattie(validManifest, ['head.ply', 'states.json', 'manifest.json'], '0.1.1');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('missing-files');
    expect(result.message).toContain('bone_tree.json');
  });
});
