const path = require("path");
const fs = require("fs");
const os = require("os");
const stream = require("stream");
const unzip = require("unzip-stream");
const fetch = require("node-fetch");
const spawn = require("cross-spawn");

const macUrl =
  "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip";
const winUrl =
  "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.8.1%2Bcpu.zip";
const lnxUrl =
  "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip";

let url;
switch (os.platform()) {
  case "win32":
    url = winUrl;
    break;
  case "darwin":
    url = macUrl;
    break;
  case "linux":
    url = lnxUrl;
    break;
  default:
    throw new Error("Unsupported platform");
}

const libtorchPath = path.join(__dirname, "..", "libtorch");
const zipPath = path.join(__dirname, "..", `libtorch.${os.platform()}.zip`);

(async () => {
  console.log("[libtorchjs]: Removing decompressed folder…");
  fs.rmdirSync(libtorchPath, { recursive: true, force: true });
  console.log("[libtorchjs]: Checking for existing zip…");
  if (!fs.existsSync(zipPath)) {
    console.log("[libtorchjs]: Downloading zip…");
    await downloadFile(url, zipPath);
  }
  console.log("[libtorchjs]: Extracting zip…");
  await new Promise((resolve, reject) => {
    stream.pipeline(
      fs.createReadStream(zipPath),
      unzip.Extract({ path: path.dirname(libtorchPath) }),
      (error) => {
        error ? reject(error) : resolve();
      }
    );
  });
  console.log("[libtorchjs]: Installing dependencies…");
  const result = spawn.sync("npm", ["install", "--ignore-scripts"], {
    cwd: path.join(__dirname, ".."),
  });
})();

// download file https://stackoverflow.com/a/51302466
async function downloadFile(url, path) {
  const res = await fetch(url);
  const fileStream = fs.createWriteStream(path);
  await new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", reject);
    fileStream.on("finish", resolve);
  });
}
