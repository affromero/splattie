import { ImageResponse } from 'next/og';

// Dynamically generated social card (1200×630) with a clear headline + CTA
// baked into the pixels. Next auto-wires this into og:image and twitter:image.
export const alt = 'Splattie: interactive rigged 3D Gaussian assets from one image';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '72px 80px',
          background: '#08080C',
          color: '#ffffff',
          fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', fontSize: 34, fontWeight: 700, color: '#7EB8F0' }}>
          splattie
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', flexDirection: 'column', fontSize: 78, fontWeight: 800, lineHeight: 1.04, letterSpacing: 0 }}>
            <span>Edit the splat.</span>
            <span style={{ color: '#C4A0F0' }}>Embed the asset.</span>
          </div>
          <div style={{ display: 'flex', fontSize: 32, color: '#9a9aae', marginTop: 22, maxWidth: 920 }}>
            Interactive rigged 3D Gaussian assets from one image. An open-source web component.
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              background: '#7EB8F0',
              color: '#08080C',
              fontSize: 30,
              fontWeight: 700,
              padding: '16px 34px',
              borderRadius: 999,
            }}
          >
            Try the editor free → splattie.app
          </div>
          <div style={{ display: 'flex', fontSize: 24, color: '#6b6b7b' }}>MIT · one line of HTML</div>
        </div>
      </div>
    ),
    { ...size }
  );
}
