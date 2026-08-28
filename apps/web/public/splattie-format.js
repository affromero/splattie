// Shared .splattie validation used by the editor (runtime, imported as
// /splattie-format.js) and by unit tests (via splattie-format.d.ts).
// Mirrors the widget's own checks in SplatWidget.connectedCallback so the
// editor can explain *why* a file was rejected instead of failing silently.

/**
 * @param {Record<string, any> | null} manifest  Parsed manifest.json, or null if absent.
 * @param {string[]} fileNames                    File entry names present in the ZIP.
 * @param {string | null} expectedVersion         Widget format version, or null to skip the check.
 * @returns {{ ok: boolean, code?: string, message?: string }}
 */
export function inspectSplattie(manifest, fileNames, expectedVersion) {
  if (!manifest) {
    return {
      ok: false,
      code: 'no-manifest',
      message:
        'This .splattie has no manifest.json - it was made with an older pipeline. ' +
        'Re-export it with the current version.',
    };
  }

  if (manifest.format !== 'splattie') {
    return {
      ok: false,
      code: 'not-splattie',
      message: `This is not a .splattie file (format: "${manifest.format ?? 'unknown'}").`,
    };
  }

  if (expectedVersion && manifest.formatVersion !== expectedVersion) {
    return {
      ok: false,
      code: 'version-mismatch',
      message:
        `Version mismatch: this file is v${manifest.formatVersion}, the editor is v${expectedVersion}. ` +
        'Re-export this .splattie with the current version.',
    };
  }

  const assetType = manifest.assetType ?? 'head';
  if (!['head', 'body', 'object'].includes(assetType)) {
    return {
      ok: false,
      code: 'unsupported-asset-type',
      message:
        `Unsupported assetType "${assetType}"; the widget only loads head, body, or object bundles. ` +
        'Re-export this .splattie with the current version.',
    };
  }

  const referenced = [
    manifest.avatar?.splat?.file,
    manifest.widget?.config,
    manifest.animation?.skeleton?.file,
    manifest.animation?.weights?.file,
  ].filter(Boolean);
  const missing = referenced.filter((name) => !fileNames.includes(name));
  if (missing.length) {
    return {
      ok: false,
      code: 'missing-files',
      message: `The .splattie is missing files its manifest references: ${missing.join(', ')}.`,
    };
  }

  return { ok: true };
}
