const modelScreen = document.getElementById("model-screen");
const dashboardScreen = document.getElementById("dashboard-screen");
const modelTitle = document.getElementById("model-title");
const changeModelBtn = document.getElementById("change-model");
const imageInput = document.getElementById("image-input");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const detectResult = document.getElementById("detect-result");
const roastResult = document.getElementById("roast-result");
const statusLine = document.getElementById("status-line");
const resultImage = document.getElementById("result-image");

const modelMap = {
  yolo26n: "모바일용 (YOLO26n)",
  yolo26s: "웹용 (YOLO26s)",
};

let selectedModel = null;

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("status-error", isError);
}

function renderDetection(result) {
  const counts = result.counts || {};
  const entries = Object.entries(counts);
  if (!entries.length) {
    detectResult.textContent = "결점두가 검출되지 않았습니다.";
    return;
  }

  const lines = entries
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `- ${name}: ${count}`)
    .join("\n");

  detectResult.textContent = `검출 클래스 수\n${lines}`;
}

function renderRoast(result) {
  const top2 = result.top2 || [];
  const alt = top2[1] ? `<br/>차순위: ${top2[1].class_name} (${(top2[1].confidence * 100).toFixed(1)}%)` : "";
  roastResult.innerHTML = `예상 로스팅 단계: <strong style='color:#2f6f4f;'>${result.predicted_stage}</strong><br/>신뢰도: ${(result.confidence * 100).toFixed(1)}%${alt}`;
}

for (const button of document.querySelectorAll(".model-btn")) {
  button.addEventListener("click", () => {
    selectedModel = button.dataset.model;
    modelTitle.textContent = modelMap[selectedModel];
    modelScreen.classList.remove("screen-active");
    dashboardScreen.classList.add("screen-active");
  });
}

changeModelBtn.addEventListener("click", () => {
  dashboardScreen.classList.remove("screen-active");
  modelScreen.classList.add("screen-active");
  setStatus("이미지를 업로드하고 분석 실행을 눌러주세요.");
});

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = "block";
    resultImage.style.display = "none";
    setStatus("이미지 업로드 완료. 분석 실행을 눌러주세요.");
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedModel) {
    alert("먼저 모델을 선택해주세요.");
    return;
  }

  if (!imageInput.files.length) {
    alert("먼저 이미지를 업로드해주세요.");
    return;
  }

  try {
    analyzeBtn.disabled = true;
    setStatus("모델 추론 중입니다. 잠시만 기다려주세요...");

    const formData = new FormData();
    formData.append("model", selectedModel);
    formData.append("image", imageInput.files[0]);

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "분석 요청에 실패했습니다.");
    }

    renderDetection(payload.detection);
    renderRoast(payload.roast);
    resultImage.src = payload.detection.annotated_image;
    resultImage.style.display = "block";
    setStatus(`분석 완료 (${payload.device})`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    analyzeBtn.disabled = false;
  }
});
