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
    document.addEventListener('pointerup', this.onPointerUp);
    document.addEventListener('pointercancel', this.onPointerUp);
  }

  detach(): void {
    document.removeEventListener('pointermove', this.onMove);
    document.removeEventListener('pointerleave', this.onLeave);
    document.removeEventListener('pointerup', this.onPointerUp);
    document.removeEventListener('pointercancel', this.onPointerUp);
    this.element = null;
  }

  private onMove = (e: PointerEvent): void => {
    if (!this.element) return;
    const rect = this.element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    this.ndcX = (e.clientX - centerX) / (rect.width / 2);
    this.ndcY = -((e.clientY - centerY) / (rect.height / 2));

    this.clientX = e.clientX - rect.left;
    this.clientY = e.clientY - rect.top;
    this.isOnPage = true;
  };

  private onLeave = (): void => {
    this.isOnPage = false;
    this.ndcX = 0;
    this.ndcY = 0;
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') {
      this.isOnPage = false;
      this.ndcX = 0;
      this.ndcY = 0;
    }
  };
}
