export class SplatEvents {
  private element: HTMLElement;
  private wasOnSplat = false;

  constructor(element: HTMLElement) {
    this.element = element;
  }

  update(isOnSplat: boolean): void {
    if (isOnSplat && !this.wasOnSplat) {
      this.element.dispatchEvent(new CustomEvent('splathover', { bubbles: true }));
    }
    if (!isOnSplat && this.wasOnSplat) {
      this.element.dispatchEvent(new CustomEvent('splatleave', { bubbles: true }));
    }
    this.wasOnSplat = isOnSplat;
  }

  attachClick(element: HTMLElement): void {
    element.addEventListener('click', () => {
      if (this.wasOnSplat) {
        this.element.dispatchEvent(new CustomEvent('splatclick', { bubbles: true }));
      }
    });
    element.addEventListener('dblclick', () => {
      if (this.wasOnSplat) {
        this.element.dispatchEvent(new CustomEvent('splatdblclick', { bubbles: true }));
      }
    });
  }
}
