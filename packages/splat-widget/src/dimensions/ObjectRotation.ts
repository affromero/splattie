const DEG2RAD = Math.PI / 180;

export class ObjectRotation {
  apply(
    mesh: { rotation: { set: (x: number, y: number, z: number) => void } },
    rotation: [number, number, number],
  ): void {
    mesh.rotation.set(
      rotation[0] * DEG2RAD,
      rotation[1] * DEG2RAD,
      rotation[2] * DEG2RAD,
    );
  }
}
