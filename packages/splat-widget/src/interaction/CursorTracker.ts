export class CursorTracker {
  ndcX = 0;
  ndcY = 0;
  clientX = 0;
  clientY = 0;
  isOnPage = false;

  attach(element: HTMLElement): void {
    element.addEventListener('pointermove', this.onMove);
    element.addEventListener('pointerleave', this.onLeave);
  }

  detach(element: HTMLElement): void {
    element.removeEventListener('pointermove', this.onMove);
    element.removeEventListener('pointerleave', this.onLeave);
  }

  private onMove = (e: PointerEvent): void => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    this.clientX = e.clientX - rect.left;
    this.clientY = e.clientY - rect.top;
    this.ndcX = (this.clientX / rect.width) * 2 - 1;
    this.ndcY = -((this.clientY / rect.height) * 2 - 1);
    this.isOnPage = true;
  };

  private onLeave = (): void => {
    this.isOnPage = false;
    this.ndcX = 0;
    this.ndcY = 0;
  };
}
