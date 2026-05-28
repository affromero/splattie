export type SplattieInspectionCode =
  | 'no-manifest'
  | 'not-splattie'
  | 'version-mismatch'
  | 'missing-files';

export interface SplattieInspection {
  ok: boolean;
  code?: SplattieInspectionCode;
  message?: string;
}

export function inspectSplattie(
  manifest: Record<string, unknown> | null,
  fileNames: string[],
  expectedVersion: string | null,
): SplattieInspection;
