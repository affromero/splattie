export class Expression {
  apply(target: Record<string, number>, weights: Record<string, number>): void {
    for (const [key, value] of Object.entries(weights)) {
      target[key] = value;
    }
  }
}
