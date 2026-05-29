// Privacy-first country resolution: derived entirely client-side from the
// browser's locale + IANA timezone (never from the IP). Returns an ISO 3166-1
// alpha-2 code, or '' when it can't be determined.

// Compact IANA timezone -> country map covering the most common zones. Not
// exhaustive by design; the locale region is tried first and covers the rest.
const TZ_COUNTRY: Record<string, string> = {
  'America/New_York': 'US', 'America/Chicago': 'US', 'America/Denver': 'US',
  'America/Los_Angeles': 'US', 'America/Phoenix': 'US', 'America/Anchorage': 'US',
  'America/Toronto': 'CA', 'America/Vancouver': 'CA', 'America/Edmonton': 'CA',
  'America/Mexico_City': 'MX', 'America/Monterrey': 'MX', 'America/Tijuana': 'MX',
  'America/Bogota': 'CO', 'America/Lima': 'PE', 'America/Santiago': 'CL',
  'America/Sao_Paulo': 'BR', 'America/Fortaleza': 'BR', 'America/Manaus': 'BR',
  'America/Argentina/Buenos_Aires': 'AR', 'America/Montevideo': 'UY',
  'America/Caracas': 'VE', 'America/Guayaquil': 'EC', 'America/La_Paz': 'BO',
  'America/Asuncion': 'PY', 'America/Panama': 'PA', 'America/Costa_Rica': 'CR',
  'America/Guatemala': 'GT', 'America/Santo_Domingo': 'DO', 'America/Havana': 'CU',
  'Europe/London': 'GB', 'Europe/Dublin': 'IE', 'Europe/Lisbon': 'PT',
  'Europe/Madrid': 'ES', 'Europe/Paris': 'FR', 'Europe/Brussels': 'BE',
  'Europe/Amsterdam': 'NL', 'Europe/Berlin': 'DE', 'Europe/Zurich': 'CH',
  'Europe/Vienna': 'AT', 'Europe/Rome': 'IT', 'Europe/Warsaw': 'PL',
  'Europe/Prague': 'CZ', 'Europe/Stockholm': 'SE', 'Europe/Oslo': 'NO',
  'Europe/Copenhagen': 'DK', 'Europe/Helsinki': 'FI', 'Europe/Athens': 'GR',
  'Europe/Bucharest': 'RO', 'Europe/Budapest': 'HU', 'Europe/Kyiv': 'UA',
  'Europe/Kiev': 'UA', 'Europe/Moscow': 'RU', 'Europe/Istanbul': 'TR',
  'Asia/Jerusalem': 'IL', 'Asia/Dubai': 'AE', 'Asia/Riyadh': 'SA',
  'Asia/Karachi': 'PK', 'Asia/Kolkata': 'IN', 'Asia/Calcutta': 'IN',
  'Asia/Dhaka': 'BD', 'Asia/Bangkok': 'TH', 'Asia/Jakarta': 'ID',
  'Asia/Ho_Chi_Minh': 'VN', 'Asia/Manila': 'PH', 'Asia/Kuala_Lumpur': 'MY',
  'Asia/Singapore': 'SG', 'Asia/Hong_Kong': 'HK', 'Asia/Taipei': 'TW',
  'Asia/Shanghai': 'CN', 'Asia/Tokyo': 'JP', 'Asia/Seoul': 'KR',
  'Australia/Sydney': 'AU', 'Australia/Melbourne': 'AU', 'Australia/Perth': 'AU',
  'Pacific/Auckland': 'NZ', 'Africa/Johannesburg': 'ZA', 'Africa/Lagos': 'NG',
  'Africa/Nairobi': 'KE', 'Africa/Cairo': 'EG', 'Africa/Casablanca': 'MA',
};

/** Best-effort ISO country code from locale region, then timezone. '' if unknown. */
export function resolveCountry(): string {
  if (typeof navigator === 'undefined') return '';
  try {
    const locale = navigator.languages?.[0] || navigator.language || '';
    const region = locale.match(/-([A-Za-z]{2})(?:$|-)/);
    if (region) return region[1].toUpperCase();
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return TZ_COUNTRY[tz] ?? '';
  } catch {
    return '';
  }
}
