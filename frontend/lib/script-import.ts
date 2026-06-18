import type { ScriptInput } from "./api";

export function titleFromScriptFileName(fileName: string) {
  const baseName = fileName.replace(/\\/g, "/").split("/").pop()?.trim() ?? "";
  const title = baseName.replace(/\.[^/.]+$/, "").replace(/\s+/g, " ").trim();
  return title || "Uploaded script";
}

export async function scriptInputFromFile(file: File): Promise<ScriptInput> {
  const text = (await file.text()).replace(/\r\n/g, "\n").trim();
  if (!text) {
    throw new Error(`${file.name || "Uploaded file"} is empty.`);
  }
  return {
    title: titleFromScriptFileName(file.name),
    text,
    is_published: false,
  };
}
