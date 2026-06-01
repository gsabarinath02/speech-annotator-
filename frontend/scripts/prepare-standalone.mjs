import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const standaloneRoot = join(".next", "standalone");
const standaloneNext = join(standaloneRoot, ".next");
const standaloneStatic = join(standaloneNext, "static");
const standalonePublic = join(standaloneRoot, "public");

mkdirSync(standaloneNext, { recursive: true });

if (existsSync(standaloneStatic)) {
  rmSync(standaloneStatic, { recursive: true, force: true });
}
cpSync(join(".next", "static"), standaloneStatic, { recursive: true });

if (existsSync("public")) {
  if (existsSync(standalonePublic)) {
    rmSync(standalonePublic, { recursive: true, force: true });
  }
  cpSync("public", standalonePublic, { recursive: true });
}
