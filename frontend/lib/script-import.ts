import type { ScriptInput } from "./api";

export type ScriptImportItem = ScriptInput & {
  id: string;
  fileName: string;
  lineCount: number;
};

const SCRIPT_FILE_PATTERN = /\.(txt|md)$/i;
const SPLIT_CATEGORY_ORDER = [
  ["introduction", "verification", "privacy"],
  ["device_setup", "tablet"],
  ["device_setup", "blood_pressure_monitor"],
  ["device_setup", "pulse_oximeter"],
  ["device_setup", "scale"],
  ["device_setup", "thermometer"],
  ["troubleshoot", "tablet"],
  ["troubleshoot", "blood_pressure_monitor"],
  ["troubleshoot", "pulse_oximeter"],
  ["troubleshoot", "scale"],
  ["troubleshoot", "thermometer"],
  ["faq", "escalation", "support", "closing"],
];

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

export async function scriptImportItemsFromFiles(files: File[]): Promise<ScriptImportItem[]> {
  const scriptFiles = files.filter((file) => SCRIPT_FILE_PATTERN.test(file.name)).sort(compareScriptFiles);
  const inputs = await Promise.all(scriptFiles.map((file) => scriptInputFromFile(file)));
  return inputs.map((input, index) => ({
    ...input,
    id: `${scriptFiles[index].name}-${index}`,
    fileName: scriptFiles[index].name,
    lineCount: input.text.split("\n").filter((line) => line.trim()).length,
  }));
}

export function moveScriptImportItem(items: ScriptImportItem[], fromIndex: number, toIndex: number) {
  if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0 || fromIndex >= items.length || toIndex >= items.length) {
    return items;
  }
  const nextItems = [...items];
  const [movedItem] = nextItems.splice(fromIndex, 1);
  nextItems.splice(toIndex, 0, movedItem);
  return nextItems;
}

function compareScriptFiles(firstFile: File, secondFile: File) {
  const firstTitle = titleFromScriptFileName(firstFile.name);
  const secondTitle = titleFromScriptFileName(secondFile.name);
  const firstSequence = splitSequence(firstTitle);
  const secondSequence = splitSequence(secondTitle);

  return (
    firstSequence.sourceId - secondSequence.sourceId ||
    firstSequence.categoryRank - secondSequence.categoryRank ||
    firstTitle.localeCompare(secondTitle, undefined, { numeric: true, sensitivity: "base" })
  );
}

function splitSequence(title: string) {
  const normalizedTitle = normalizeTitle(title);
  const sourceId = Number(normalizedTitle.match(/(?:^|_)(\d{3,})(?:_|$)/)?.[1] ?? Number.MAX_SAFE_INTEGER);
  const categoryRank = SPLIT_CATEGORY_ORDER.findIndex((tokens) => tokens.every((token) => normalizedTitle.includes(token)));
  return {
    sourceId,
    categoryRank: categoryRank === -1 ? Number.MAX_SAFE_INTEGER : categoryRank,
  };
}

function normalizeTitle(title: string) {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}
