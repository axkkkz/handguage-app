// ตัวจับจุดมือบนเว็บ: MediaPipe Hand Landmarker (รุ่นเว็บ) ครอบให้ฝั่ง Dart เรียกง่ายๆ
//
// ภาพจากกล้องถูกประมวลผลในเบราว์เซอร์นี้เท่านั้น ไม่ถูกส่งออกไปไหน
// ไลบรารีและโมเดลโหลดจาก CDN ตอนเปิดกล้องครั้งแรก (ไม่โหลดตอนเปิดแอพ)
//
// ใช้จาก lib/recognition/device_hand_source_web.dart:
//   window.hgHandTracker.start(videoElement, onFrame)  -> Promise<เลขรอบ>
//   window.hgHandTracker.stop(เลขรอบ)  หยุดเฉพาะรอบนั้น (ไม่ส่ง = หยุดทุกรอบ)
//   window.hgHandTracker.stats()                       -> ข้อความสำหรับโหมดดีบัก
// onFrame ได้ Array 42 ค่าต่อมือ [x0, y0, x1, y1, ...] (สองมือ = 84) หน่วยพิกเซลของวิดีโอ หรือ null ถ้าไม่เห็นมือ
(function () {
  const VERSION = "0.10.14";
  const BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
  const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
  const MODEL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  // จับมือไม่เกินประมาณ 12 ครั้ง/วินาที: การจับมือรันบนเธรดเดียวกับหน้าจอ
  // ถ้าทำทุกเฟรมที่จอวาด (60 ครั้ง/วินาที) มือถือจะกระตุกและร้อน ส่วนตัวจำท่าต้องการแค่
  // เฟรมนิ่งต่อเนื่องไม่กี่เฟรม
  const MIN_INTERVAL_MS = 80;

  // โหลดไลบรารี+โมเดล (ราว 8 MB) ไม่เกินนี้ เน็ตมือถือช้าก็ยังทัน
  const LOAD_TIMEOUT_MS = 45000;

  let landmarker = null;
  let loading = null;
  let delegateUsed = "";
  // เลขรอบปัจจุบัน: หน้ากล้องเก่าที่ปิดช้าเรียก stop ด้วยเลขรอบเก่า จะไม่ไปหยุดรอบของหน้าใหม่
  let session = 0;
  let frameRequest = 0;
  let avgDetectMs = 0;
  let framesPerSecond = 0;

  async function create(vision, delegate) {
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM);
    return vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2, // ท่าสองมือ (ตัวจำท่านิ่งใช้มือหลักมือเดียว)
    });
  }

  function ensureLoaded() {
    if (landmarker) return Promise.resolve(landmarker);
    if (!loading) {
      loading = (async () => {
        const vision = await import(BUNDLE);
        try {
          landmarker = await create(vision, "GPU");
          delegateUsed = "GPU";
        } catch (_) {
          // บางเครื่องใช้ GPU ไม่ได้: ถอยไปใช้ CPU
          landmarker = await create(vision, "CPU");
          delegateUsed = "CPU";
        }
        return landmarker;
      })();
      // CDN ค้าง (ไม่ตอบแต่ไม่ล้ม): ไม่รอเกิน LOAD_TIMEOUT_MS ให้หน้าจอขึ้นข้อผิดพลาดและกดลองใหม่ได้
      // (ถ้าโหลดเสร็จทีหลัง landmarker ก็ยังถูกเก็บไว้ใช้รอบถัดไป)
      const timeout = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("hand tracker load timed out")), LOAD_TIMEOUT_MS));
      loading = Promise.race([loading, timeout]).catch((e) => {
        loading = null; // ให้ลองใหม่ได้ในครั้งถัดไป
        throw e;
      });
    }
    return loading;
  }

  let lastTimestamp = 0;

  /** จับจุดมือจากเฟรมปัจจุบันของ [video] คืน Array 42 ค่าต่อมือ (สองมือ = 84) หรือ null */
  function detect(video, timestampMs) {
    // ภาพยังไม่มีขนาด (กล้องเพิ่งเปิด/สลับ): ข้าม ถ้าส่งเข้าไป MediaPipe จะ error
    // และหลังจากนั้นทุกเฟรมจะ error ตาม (ตัวจับมือพังทั้งตัว)
    if (!video.videoWidth || !video.videoHeight) return null;
    // โหมด VIDEO ต้องการเวลาที่เพิ่มขึ้นเสมอ
    const ts = Math.max(timestampMs, lastTimestamp + 1);
    lastTimestamp = ts;
    const result = landmarker.detectForVideo(video, ts);
    const hands = (result.landmarks || []).filter((hand) => hand && hand.length === 21);
    if (!hands.length) return null;
    // ค่าที่ได้เป็นสัดส่วน 0..1 ของภาพ ต้องคูณขนาดจริง ไม่งั้นรูปมือเพี้ยนตามอัตราส่วนภาพ
    const w = video.videoWidth;
    const h = video.videoHeight;
    const out = [];
    for (const hand of hands.slice(0, 2)) {
      for (let i = 0; i < 21; i++) out.push(hand[i].x * w, hand[i].y * h);
    }
    return out;
  }

  let failures = 0;

  /** ทิ้งตัวจับมือที่พังแล้วสร้างใหม่ ระหว่างสร้าง ลูปหลักรอ (ไม่จับมือ) */
  function rebuild() {
    const broken = landmarker;
    landmarker = null;
    loading = null;
    lastTimestamp = 0;
    try {
      broken && broken.close();
    } catch (_) {}
    rebuilds++;
    ensureLoaded().catch(() => {});
  }

  let rebuilds = 0;

  function waitForFrame(video) {
    if (video.readyState >= 2) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        video.removeEventListener("loadeddata", done);
        resolve();
      };
      video.addEventListener("loadeddata", done);
      setTimeout(done, 3000); // ไม่รอเกินนี้ ลูปหลักจะข้ามเฟรมที่ยังไม่พร้อมเอง
    });
  }

  window.hgHandTracker = {
    ensureLoaded,
    detect,

    /** เช่น "GPU 28 ms 11/s" = ตัวเร่ง, เวลาจับมือเฉลี่ยต่อเฟรม, จำนวนครั้งต่อวินาที */
    stats() {
      if (!landmarker) return rebuilds ? `rebuilding (${rebuilds})` : "";
      const r = rebuilds ? ` rebuilt ${rebuilds}` : "";
      return `${delegateUsed} ${Math.round(avgDetectMs)} ms ${framesPerSecond}/s${r}`;
    },

    async start(video, onFrame) {
      const my = ++session;
      cancelAnimationFrame(frameRequest);
      await ensureLoaded();
      // การจับมือครั้งแรกช้ากว่าปกติมาก (เตรียมตัวเร่ง) ทำให้เสร็จตรงนี้ ระหว่างที่หน้าจอ
      // ยังแสดงสถานะกำลังเปิดกล้อง ไม่ให้ไปค้างตอนผู้ใช้เริ่มทำท่าแล้ว
      await waitForFrame(video);
      try {
        detect(video, performance.now());
      } catch (_) {}

      if (my !== session) return my; // มีรอบใหม่เริ่มแทนระหว่างรอโหลด
      avgDetectMs = 0;
      let lastTime = -1;
      let nextAt = 0;
      let counted = 0;
      let countStart = performance.now();
      let lastPlayTry = 0;
      const loop = () => {
        if (my !== session) return;
        const now = performance.now();
        // Safari หยุดวิดีโอเองเมื่อมันถูกย้ายในหน้า (Flutter วาดหน้าใหม่) แล้วไม่เล่นต่อ
        // ภาพจึงค้าง/ดำ: สั่งเล่นต่อเอง ไม่เกินครึ่งวินาทีครั้ง
        if (video.paused && now - lastPlayTry > 500) {
          lastPlayTry = now;
          video.play().catch(() => {});
        }
        // ประมวลผลเฉพาะเมื่อถึงเวลา มีเฟรมใหม่ และตัวจับมือพร้อม (อาจกำลังสร้างใหม่)
        if (landmarker && now >= nextAt && video.readyState >= 2 && video.currentTime !== lastTime) {
          lastTime = video.currentTime;
          let points = null;
          try {
            points = detect(video, now);
            failures = 0;
          } catch (_) {
            // เฟรมเดียวพลาดไม่เป็นไร แต่ถ้าพลาดติดกัน แปลว่าตัวจับมือพังทั้งตัว
            // (MediaPipe ไม่ฟื้นเอง) จึงทิ้งแล้วสร้างใหม่
            failures++;
            if (failures >= 3) {
              failures = 0;
              rebuild();
            }
          }
          const took = performance.now() - now;
          avgDetectMs = avgDetectMs ? avgDetectMs * 0.8 + took * 0.2 : took;
          // เครื่องช้า: เว้นให้หน้าจอได้ทำงานอย่างน้อยเท่ากับเวลาที่ใช้จับมือ
          nextAt = now + Math.max(MIN_INTERVAL_MS, took * 2);
          counted++;
          if (now - countStart >= 1000) {
            framesPerSecond = Math.round((counted * 1000) / (now - countStart));
            counted = 0;
            countStart = now;
          }
          try {
            onFrame(points);
          } catch (_) {}
        }
        frameRequest = requestAnimationFrame(loop);
      };
      loop();
      return my;
    },

    stop(id) {
      if (id !== undefined && id !== null && id !== session) return;
      session++;
      cancelAnimationFrame(frameRequest);
      // หยุดแล้ว: ข้อมูลวินิจฉัยต้องไม่โชว์จำนวนภาพ/วินาทีของรอบเก่าค้างไว้
      framesPerSecond = 0;
    },
  };
})();
