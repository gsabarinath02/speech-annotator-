export function nextScriptIndexAfterSave(currentIndex: number, scriptCount: number) {
  if (scriptCount <= 0) return 0;
  return Math.min(Math.max(currentIndex, 0) + 1, scriptCount - 1);
}
