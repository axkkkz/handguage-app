// ตัวจับจุดมือบนเว็บ: MediaPipe Hand Landmarker (รุ่นเว็บ) ครอบให้ฝั่ง Dart เรียกง่ายๆ
//
// ภาพจากกล้องถูกประมวลผลในเบราว์เซอร์นี้เท่านั้น ไม่ถูกส่งออกไปไหน
// ไลบรารีและโมเดลโหลดจาก CDN ตอนเปิดกล้องครั้งแรก (ไม่โหลดตอนเปิดแอพ)
//
// ใช้จาก lib/recognition/device_hand_source_web.dart:
//   window.hgHandTracker.start(videoElement, onFrame)  -> Promise
//   window.hgHandTracker.stop()
// onFrame ได้ Array 42 ค่า [x0, y0, x1, y1, ...] หน่วยพิกเซลของวิดีโอ หรือ null ถ้าไม่เห็นมือ
(function () {
  const VERSION = "0.10.14";
  const BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
  const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
  const MODEL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  let landmarker = null;
  let loading = null;
  let running = false;
  let frameRequest = 0;

  async function create(vision, delegate) {
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM);
    return vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 1,
    });
  }

  function ensureLoaded() {
    if (landmarker) return Promise.resolve(landmarker);
    if (!loading) {
      loading = (async () => {
        const vision = await import(BUNDLE);
        try {
          landmarker = await create(vision, "GPU");
        } catch (_) {
          // บางเครื่องใช้ GPU ไม่ได้: ถอยไปใช้ CPU
          landmarker = await create(vision, "CPU");
        }
        return landmarker;
      })().catch((e) => {
        loading = null; // ให้ลองใหม่ได้ในครั้งถัดไป
        throw e;
      });
    }
    return loading;
  }

  /** จับจุดมือจากเฟรมปัจจุบันของ [video] คืน Array 42 ค่า หรือ null */
  function detect(video, timestampMs) {
    const result = landmarker.detectForVideo(video, timestampMs);
    const hand = result.landmarks && result.landmarks[0];
    if (!hand || hand.length !== 21) return null;
    // ค่าที่ได้เป็นสัดส่วน 0..1 ของภาพ ต้องคูณขนาดจริง ไม่งั้นรูปมือเพี้ยนตามอัตราส่วนภาพ
    const w = video.videoWidth;
    const h = video.videoHeight;
    const out = new Array(42);
    for (let i = 0; i < 21; i++) {
      out[2 * i] = hand[i].x * w;
      out[2 * i + 1] = hand[i].y * h;
    }
    return out;
  }

  window.hgHandTracker = {
    ensureLoaded,
    detect,

    async start(video, onFrame) {
      await ensureLoaded();
      running = true;
      let lastTime = -1;
      const loop = () => {
        if (!running) return;
        // ประมวลผลเฉพาะเมื่อมีเฟรมใหม่
        if (video.readyState >= 2 && video.currentTime !== lastTime) {
          lastTime = video.currentTime;
          try {
            onFrame(detect(video, performance.now()));
          } catch (_) {
            // เฟรมเดียวพลาดไม่เป็นไร
          }
        }
        frameRequest = requestAnimationFrame(loop);
      };
      loop();
    },

    stop() {
      running = false;
      cancelAnimationFrame(frameRequest);
    },
  };
})();
