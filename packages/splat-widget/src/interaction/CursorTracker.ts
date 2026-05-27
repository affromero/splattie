export class CursorTracker {
  ndcX = 0;
  ndcY = 0;
  clientX = 0;
  clientY = 0;
  isOnPage = false;
  private element: HTMLElement | null = null;

  attach(element: HTMLElement): void {
    this.element = element;
    document.addEventListener('pointermove', this.onMove);
    document.addEventListener('pointerleave', this.onLeave);
  }

  detach(): void {
    document.removeEventListener('pointermove', this.onMove);
    document.removeEventListener('pointerleave', this.onLeave);
    this.element = null;
  }

  private onMove = (e: PointerEvent): void => {
    if (!this.element) return;
    const rect = this.element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    // NDC relative to widget center, but extends beyond -1..+1 when cursor is outside
    this.ndcX = (e.clientX - centerX) / (rect.width / 2);
    this.ndcY = -((e.clientY - centerY) / (rect.height / 2));

    // Client coords relative to widget for hit detection
    this.clientX = e.clientX - rect.left;
    this.clientY = e.clientY - rect.top;
    this.isOnPage = true;
  };

  private onLeave = (): void => {
    this.isOnPage = false;
    this.ndcX = 0;
    this.ndcY = 0;
  };
}
