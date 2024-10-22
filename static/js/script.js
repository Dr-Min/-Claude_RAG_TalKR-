// 모듈 패턴을 사용하여 전역 네임스페이스 오염 방지
const ChatApp = (function () {
  // 비공개 변수
  let isLoggedIn = false;
  let isListening = false;
  let isAITalking = false;
  let isLoading = false;
  let isAutoMicOn = false;
  let currentAudio = null;
  let recognition = null;
  let messageCount = 0;
  let sessionStartTime = null;
  let isTranslating = false;
  let pendingMessage = null;
  let messageQueue = [];

  // DOM 요소 캐싱
  const elements = {
    chatContainer: document.getElementById("chat-container"),
    userInput: document.getElementById("user-input"),
    sendBtn: document.getElementById("send-btn"),
    voiceBtn: document.getElementById("voice-btn"),
    autoMicToggle: document.getElementById("auto-mic-toggle"),
    authModal: document.getElementById("auth-modal"),
    loginBtn: document.getElementById("login-btn"),
    signupBtn: document.getElementById("signup-btn"),
    authMessage: document.getElementById("auth-message"),
    modalTitle: document.getElementById("modal-title"),
    loginForm: document.getElementById("login-form"),
    signupForm: document.getElementById("signup-form"),
    showSignupLink: document.getElementById("show-signup"),
    showLoginLink: document.getElementById("show-login"),
    menuIcon: document.getElementById("menu-icon"),
    sidebar: document.getElementById("sidebar"),
    closeSidebar: document.getElementById("close-sidebar"),
    userId: document.getElementById("user-id"),
    showHistory: document.getElementById("show-history"),
    historyModal: document.getElementById("history-modal"),
    closeHistory: document.getElementById("close-history"),
    historyContainer: document.getElementById("history-container"),
    loadingHistory: document.getElementById("loading-history"),
    showForgotPasswordLink: document.getElementById("show-forgot-password"),
    forgotPasswordForm: document.getElementById("forgot-password-form"),
    backToLoginLink: document.getElementById("back-to-login"),
    resetPasswordBtn: document.getElementById("reset-password-btn"),
    // showReports: document.getElementById("show-reports"),
    // reportsModal: document.getElementById("reports-modal"),
    // closeReports: document.getElementById("close-reports"),
    // reportsContainer: document.getElementById("reports-container"),
    // showVocabulary: document.getElementById("show-vocabulary"),
    // vocabularyModal: document.getElementById("vocabulary-modal"),
    // closeVocabulary: document.getElementById("close-vocabulary"),
    // vocabularyContainer: document.getElementById("vocabulary-container"),
    logoutBtn: document.getElementById("logout-btn"),
    showTodaysNews: document.getElementById("show-todays-news"),
  };

  // 초기화 함수
  function init() {
    if (!elements.chatContainer) {
      console.error("Critical element is missing. Chat container not found.");
      return;
    }
    setupEventListeners();
    setupSpeechRecognition();
    checkLoginStatus();
  }

  // 이벤트 리스너 설정
  function setupEventListeners() {
    elements.sendBtn?.addEventListener("click", sendMessage);
    elements.userInput?.addEventListener("keypress", handleKeyPress);
    elements.voiceBtn?.addEventListener("click", toggleVoiceRecognition);
    elements.autoMicToggle?.addEventListener("click", toggleAutoMic);
    elements.loginBtn?.addEventListener("click", login);
    elements.signupBtn?.addEventListener("click", signup);
    elements.showSignupLink?.addEventListener("click", showSignupForm);
    elements.showLoginLink?.addEventListener("click", showLoginForm);
    elements.menuIcon?.addEventListener("click", openSidebar);
    elements.closeSidebar?.addEventListener("click", closeSidebar);
    elements.showHistory?.addEventListener("click", showHistoryModal);
    elements.closeHistory?.addEventListener("click", closeHistoryModal);
    elements.showForgotPasswordLink?.addEventListener(
      "click",
      showForgotPasswordForm
    );
    elements.backToLoginLink?.addEventListener("click", backToLogin);
    elements.resetPasswordBtn?.addEventListener("click", resetPassword);
    // elements.showReports?.addEventListener("click", showReportsModal);
    // elements.closeReports?.addEventListener("click", closeReportsModal);
    // elements.showVocabulary?.addEventListener("click", showVocabularyModal);
    // elements.closeVocabulary?.addEventListener("click", closeVocabularyModal);
    elements.logoutBtn?.addEventListener("click", logout);
    elements.showTodaysNews?.addEventListener("click", showTodaysNews);
    elements.sendBtn?.addEventListener("click", sendMessage);
    elements.userInput?.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        sendMessage(e);
      }
    });
  }

  // 메시지 전송 함수
  function sendMessage(event) {
    // event 매개변수가 존재하면 기본 동작을 막습니다.
    if (event && event.preventDefault) {
      event.preventDefault();
    }

    const message = elements.userInput.value.trim();
    if (message) {
      // 메시지가 비어있지 않은 경우에만 처리
      if (isProcessing()) {
        pendingMessage = message;
        showPendingMessageNotification();
      } else {
        messageQueue.push(message);
        processMessageQueue();
        messageCount++;
      }
      elements.userInput.value = "";
    }
  }

  function handleKeyPress(e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  }

  // 메시지 큐 처리
  function processMessageQueue() {
    if (isProcessing() || messageQueue.length === 0) {
      return;
    }

    setProcessing(true);
    const message = messageQueue.shift();
    addMessage(message, true);

    const loadingDiv = addLoadingAnimation();
    setLoading(true);
    setAITalking(true);
    stopListening();

    sendMessageToServer(message)
      .then((data) => {
        if (data.success) {
          addMessage(data.message, false, data.audio);
        } else {
          throw new Error("서버에서 오류 응답을 받았습니다.");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addMessage("네트워크 오류가 발생했습니다. 다시 시도해 주세요.", false);
      })
      .finally(() => {
        removeLoadingAnimation(loadingDiv);
        setLoading(false);
        setAITalking(false);
        setProcessing(false);
        if (pendingMessage) {
          showPendingMessageConfirmation();
        } else {
          processMessageQueue();
        }
        if (isAutoMicOn && !isAITalking) {
          startListening();
        }
      });
  }

  // 서버에 메시지 전송
  function sendMessageToServer(message) {
    return fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
    })
      .then((response) => {
        if (!response.ok)
          throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (data.timing) {
          console.log("=== 처리 시간 분석 ===");
          const sortedTiming = Object.entries(data.timing).sort((a, b) => {
            const aNum = parseInt(a[0].split("_")[0]);
            const bNum = parseInt(b[0].split("_")[0]);
            return aNum - bNum;
          });
          for (const [단계, 소요시간] of sortedTiming) {
            console.log(`${단계}: ${소요시간.toFixed(3)}초`);
          }
          console.log("===================");
        }
        return data;
      });
  }

  // 메시지 추가
  function addMessage(message, isUser, audioData) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${isUser ? "user-message" : "bot-message"}`;

    const messageBubble = document.createElement("div");
    messageBubble.className = "message-bubble";
    messageBubble.textContent = message;
    messageDiv.appendChild(messageBubble);

    if (!isUser) {
      const translateBtn = document.createElement("button");
      translateBtn.className = "translate-btn";
      translateBtn.textContent = "Translate";
      translateBtn.onclick = () =>
        translateMessage(message, messageDiv, translateBtn);
      messageDiv.appendChild(translateBtn);
    }

    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;

    if (!isUser && audioData) {
      playAudio(audioData);
    }
  }

  // 오디오 재생
  function playAudio(audioData) {
    setAITalking(true);
    if (isListening) {
      stopListening();
    }
    currentAudio = new Audio("data:audio/mp3;base64," + audioData);
    currentAudio.play().catch((error) => {
      console.error("오디오 재생 오류:", error);
      setAITalking(false);
      if (isAutoMicOn) {
        startListening();
      }
    });
    currentAudio.onended = () => {
      currentAudio = null;
      setAITalking(false);
      if (isAutoMicOn) {
        startListening();
      }
    };
  }

  // 음성 인식 설정
  function setupSpeechRecognition() {
    if (
      !("webkitSpeechRecognition" in window) &&
      !("SpeechRecognition" in window)
    ) {
      console.error("음성 인식이 지원되지 않는 브라우저입니다.");
      return;
    }

    recognition = new (window.SpeechRecognition ||
      window.webkitSpeechRecognition)();
    recognition.lang = "ko-KR";
    recognition.interimResults = true;
    recognition.continuous = true;

    recognition.onstart = () => {
      console.log("음성 인식이 시작되었습니다.");
      setListening(true);
      elements.voiceBtn.classList.add("active", "voice-active");
    };

    recognition.onend = () => {
      console.log("음성 인식이 종료되었습니다.");
      setListening(false);
      elements.voiceBtn.classList.remove("active", "voice-active");

      if (
        elements.userInput.value.trim() !== "" &&
        elements.userInput.value.trim() !== lastProcessedResult
      ) {
        lastProcessedResult = elements.userInput.value.trim();
        sendMessage(lastProcessedResult, true);
      }

      if (isAutoMicOn && !isAITalking && !isLoading) {
        startListening();
      }
    };

    recognition.onresult = handleSpeechResult;
    recognition.onerror = handleSpeechError;
  }

  // 음성 인식 결과 처리
  function handleSpeechResult(event) {
    let currentTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        currentTranscript += event.results[i][0].transcript + " ";
      }
    }

    elements.userInput.value = currentTranscript.trim();

    if (currentTranscript.trim() !== lastProcessedResult.trim()) {
      if (currentTranscript.trim() !== "") {
        lastProcessedResult = currentTranscript.trim();
        sendMessage(lastProcessedResult, true);
      }
    }
  }

  // 음성 인식 오류 처리
  function handleSpeechError(event) {
    console.error("음성 인식 오류:", event.error);
    stopListening();
    if (isAutoMicOn) {
      setTimeout(startListening, 1000);
    }
  }

  // 음성 인식 토글
  function toggleVoiceRecognition() {
    if (isAITalking || isLoading) {
      stopAITalking();
      return;
    }
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }

  // 자동 마이크 토글
  function toggleAutoMic() {
    isAutoMicOn = !isAutoMicOn;
    elements.autoMicToggle.textContent = isAutoMicOn
      ? "Auto Mic: ON"
      : "Auto Mic: OFF";
    elements.autoMicToggle.classList.toggle("active");
    if (isAutoMicOn && !isAITalking && !isLoading) {
      startListening();
    } else if (!isAutoMicOn) {
      stopListening();
    }
  }

  // 음성 인식 시작
  function startListening() {
    if (!recognition) {
      setupSpeechRecognition();
    }
    recognition.start();
    setListening(true);
    elements.voiceBtn.classList.add("active");
    console.log("음성 인식이 시작되었습니다.");
  }

  // 음성 인식 중지
  function stopListening() {
    if (recognition) {
      recognition.stop();
      setListening(false);
      elements.voiceBtn.classList.remove("active");
      console.log("음성 인식이 중지되었습니다.");
    }
  }

  // AI 발화 중지
  function stopAITalking() {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    setAITalking(false);
    setLoading(false);
    console.log("AI 발화가 중지되었습니다.");
    if (pendingMessage) {
      showPendingMessageConfirmation();
    }
  }

  // 로딩 애니메이션 추가
  function addLoadingAnimation() {
    setLoading(true);
    if (isAutoMicOn) {
      stopListening();
    }
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";

    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message-bubble loading";
    loadingDiv.innerHTML = `
      <div class="loading-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;
    messageDiv.appendChild(loadingDiv);
    elements.chatContainer.appendChild(messageDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    return messageDiv;
  }

  // 로딩 애니메이션 제거
  function removeLoadingAnimation(loadingDiv) {
    elements.chatContainer.removeChild(loadingDiv);
    setLoading(false);
    if (isAutoMicOn && !isAITalking) {
      startListening();
    }
  }

  // 번역 기능
  function translateMessage(message, messageDiv, translateBtn) {
    if (isTranslating) {
      console.log("번역이 이미 진행 중입니다.");
      return;
    }

    const existingTranslation = messageDiv.querySelector(".translation");
    if (existingTranslation) {
      existingTranslation.style.display =
        existingTranslation.style.display === "none" ? "block" : "none";
      return;
    }

    setTranslating(true);
    translateBtn.disabled = true;

    const loadingDiv = addTranslationLoadingAnimation(messageDiv);

    fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message }),
    })
      .then((response) => {
        if (!response.ok)
          throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (data.translation) {
          const translationDiv = document.createElement("div");
          translationDiv.className = "translation";
          translationDiv.textContent = data.translation;
          messageDiv.appendChild(translationDiv);
          translationDiv.style.display = "block";
          elements.chatContainer.scrollTop =
            elements.chatContainer.scrollHeight;
        } else {
          throw new Error("번역 데이터가 없습니다.");
        }
      })
      .catch((error) => {
        console.error("Translation error:", error);
        addMessage("번역 중 오류가 발생했습니다. 다시 시도해 주세요.", false);
        translateBtn.classList.remove("active");
      })
      .finally(() => {
        removeTranslationLoadingAnimation(loadingDiv);
        setTranslating(false);
        translateBtn.disabled = false;
      });
  }

  // 번역 로딩 애니메이션 추가
  function addTranslationLoadingAnimation(container) {
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "loading-animation";
    loadingDiv.innerHTML = '<div class="boxLoading"></div>';
    container.appendChild(loadingDiv);
    return loadingDiv;
  }

  // 번역 로딩 애니메이션 제거
  function removeTranslationLoadingAnimation(loadingDiv) {
    if (loadingDiv && loadingDiv.parentNode) {
      loadingDiv.parentNode.removeChild(loadingDiv);
    }
  }

  // 로그인 함수
  function login() {
    const username = document.getElementById("login-username").value;
    const password = document.getElementById("login-password").value;

    fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: username, password: password }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          setLoggedIn(true);
          elements.authModal.style.display = "none";
          updateUserId(username);
          sessionStartTime = new Date();
          startUsageTracking();
        } else {
          setMessage("Failed to log in. Please try again.", "error");
        }
      })
      .catch((error) => {
        console.error("Login error:", error);
        setMessage(
          "An error occurred while logging in. Please try again.",
          "error"
        );
      });
  }

  // 회원가입 함수
  function signup() {
    const username = document.getElementById("signup-username").value.trim();
    const email = document.getElementById("signup-email").value.trim();
    const password = document.getElementById("signup-password").value;

    if (!username || !email || !password) {
      setMessage("Please fill in all fields.", "error");
      return;
    }

    if (!isValidEmail(email)) {
      setMessage("Please enter a valid email address.", "error");
      return;
    }

    if (password.length < 4) {
      setMessage("Password must be at least 4 characters long.", "error");
      return;
    }

    fetch("/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: username,
        email: email,
        password: password,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          setMessage("Sign up successful. Please log in.", "success");
          showLoginForm();
        } else if (data.error === "username_taken") {
          setMessage(
            "Username is already taken. Please choose another.",
            "error"
          );
        } else {
          setMessage(
            "Email is already registered. Do you already have an account?",
            "error"
          );
        }
      })
      .catch((error) => {
        console.error("Signup error:", error);
        setMessage(
          "An error occurred during sign up. Please try again.",
          "error"
        );
      });
  }

  // 이메일 유효성 검사
  function isValidEmail(email) {
    const re =
      /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
  }

  // 사용 시간 추적 시작
  function startUsageTracking() {
    setInterval(() => {
      const currentTime = new Date();
      const usageTime = Math.floor((currentTime - sessionStartTime) / 1000);
      updateUsageTime(usageTime);
    }, 60000);
  }

  // 사용 시간 업데이트
  function updateUsageTime(time) {
    fetch("/update_usage_time", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ time: time }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (!data.success) {
          console.error("사용 시간 업데이트 실패");
        }
      })
      .catch((error) => {
        console.error("Usage time update error:", error);
      });
  }

  // 로그인 상태 확인
  function checkLoginStatus() {
    fetch("/check_login")
      .then((response) => response.json())
      .then((data) => {
        if (data.logged_in) {
          setLoggedIn(true);
          updateUserId(data.username);
          elements.authModal.style.display = "none";
        } else {
          showLoginForm();
        }
      })
      .catch((error) => {
        console.error("로그인 상태 확인 오류:", error);
        showLoginForm();
      });
  }

  // 로그아웃 함수
  function logout() {
    fetch("/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          setLoggedIn(false);
          showLoginForm();
          closeSidebar();
        }
      })
      .catch((error) => console.error("Logout error:", error));
  }

  // 사용자 ID 업데이트
  function updateUserId(username) {
    elements.userId.textContent = username;
  }

  // 메시지 설정
  function setMessage(message, type) {
    elements.authMessage.textContent = message;
    elements.authMessage.className = type ? type + "-message" : "";
  }

  // 로그인 폼 표시
  function showLoginForm() {
    clearAuthMessage();
    elements.modalTitle.textContent = "Login";
    elements.loginForm.style.display = "block";
    elements.signupForm.style.display = "none";
    elements.forgotPasswordForm.style.display = "none";
    elements.authModal.style.display = "block";
  }

  // 회원가입 폼 표시
  function showSignupForm(e) {
    e.preventDefault();
    clearAuthMessage();
    elements.modalTitle.textContent = "Sign Up";
    elements.loginForm.style.display = "none";
    elements.signupForm.style.display = "block";
    elements.forgotPasswordForm.style.display = "none";
  }

  // 비밀번호 재설정 폼 표시
  function showForgotPasswordForm(e) {
    e.preventDefault();
    clearAuthMessage();
    elements.loginForm.style.display = "none";
    elements.signupForm.style.display = "none";
    elements.forgotPasswordForm.style.display = "block";
    elements.modalTitle.textContent = "Reset Password";
  }

  // 로그인 폼으로 돌아가기
  function backToLogin(e) {
    e.preventDefault();
    clearAuthMessage();
    showLoginForm();
  }

  // 비밀번호 재설정
  function resetPassword() {
    const email = document.getElementById("reset-email").value;
    const loadingAnimation = document.getElementById("loading-animation");

    if (!isValidEmail(email)) {
      setMessage("Please enter a valid email address.", "error");
      return;
    }

    loadingAnimation.style.display = "block";
    elements.resetPasswordBtn.disabled = true;
    clearAuthMessage();

    fetch("/request_reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email }),
    })
      .then((response) => response.json())
      .then((data) => {
        setMessage(
          data.message,
          data.message === "Reset link sent to your email" ? "success" : "error"
        );
      })
      .catch((error) => {
        console.error("Error:", error);
        setMessage("An error occurred. Please try again.", "error");
      })
      .finally(() => {
        loadingAnimation.style.display = "none";
        elements.resetPasswordBtn.disabled = false;
      });
  }

  // 인증 메시지 초기화
  function clearAuthMessage() {
    elements.authMessage.textContent = "";
    elements.authMessage.className = "";
  }

  // 사이드바 열기
  function openSidebar() {
    elements.sidebar.style.width = "50%";
  }

  // 사이드바 닫기
  function closeSidebar() {
    elements.sidebar.style.width = "0";
  }

  // 히스토리 모달 표시
  function showHistoryModal() {
    elements.historyModal.style.display = "block";
    elements.historyContainer.innerHTML = "<p>Loading history...</p>";
    loadHistory();
  }

  // 히스토리 모달 닫기
  function closeHistoryModal() {
    elements.historyModal.style.display = "none";
  }

  // 히스토리 로드
  function loadHistory(date = null) {
    if (isLoadingHistory) return;
    setLoadingHistory(true);
    elements.loadingHistory.style.display = "block";

    fetch(`/get_history?date=${date || ""}`)
      .then((response) => response.json())
      .then((data) => {
        displayHistory(data.history);
        setLoadingHistory(false);
        elements.loadingHistory.style.display = "none";
      })
      .catch((error) => {
        console.error("Error loading history:", error);
        setLoadingHistory(false);
        elements.loadingHistory.style.display = "none";
      });
  }

  // 히스토리 표시
  function displayHistory(history) {
    elements.historyContainer.innerHTML = "";
    let currentDate = null;
    history.forEach((item) => {
      if (item.date !== currentDate) {
        currentDate = item.date;
        const dateElement = document.createElement("div");
        dateElement.className = "history-date";
        dateElement.textContent = currentDate;
        elements.historyContainer.appendChild(dateElement);
      }
      item.messages.forEach((msg) => {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${
          msg.is_user ? "user-message" : "bot-message"
        }`;
        messageDiv.innerHTML = `
          <div class="message-bubble">${msg.content}</div>
          <div class="message-time">${msg.timestamp}</div>
        `;
        elements.historyContainer.appendChild(messageDiv);
      });
    });
  }

  // 오늘의 뉴스 표시
  function showTodaysNews() {
    fetch("/get_news")
      .then((response) => response.json())
      .then((data) => {
        data.messages.forEach((message) => {
          addMessage(message, false);
        });
      })
      .catch((error) => {
        console.error("Error fetching news:", error);
        addMessage("뉴스를 가져오는 중 오류가 발생했습니다.", false);
      });
  }

  // 상태 변경 함수들
  function setLoggedIn(value) {
    isLoggedIn = value;
  }

  function setListening(value) {
    isListening = value;
  }

  function setAITalking(value) {
    isAITalking = value;
  }

  function setLoading(value) {
    isLoading = value;
  }

  function setTranslating(value) {
    isTranslating = value;
  }

  function setLoadingHistory(value) {
    isLoadingHistory = value;
  }

  function isProcessing() {
    return isLoading || isAITalking;
  }

  function setProcessing(value) {
    isLoading = value;
    isAITalking = value;
  }

  // 기 중인 메시지 알림 표시
  function showPendingMessageNotification() {
    const notification = document.createElement("div");
    notification.id = "pending-message-notification";
    notification.textContent = "대기 중인 메시지가 있습니다";
    notification.style.display = "block";
    document.body.appendChild(notification);
  }

  // 대기 중인 메시지 확인
  function showPendingMessageConfirmation() {
    if (pendingMessage) {
      if (confirm(`Do you want to send this message? "${pendingMessage}"`)) {
        sendMessage(pendingMessage);
      }
      pendingMessage = null;
      const notification = document.getElementById(
        "pending-message-notification"
      );
      if (notification) {
        notification.style.display = "none";
      }
    }
  }

  // 여기에 새 함수를 추가합니다
  function updateMessageWithAudio(messageId, audioBase64) {
    const messageElement = document.querySelector(
      `[data-message-id="${messageId}"]`
    );
    if (messageElement) {
      const audioElement = document.createElement("audio");
      audioElement.src = `data:audio/mp3;base64,${audioBase64}`;
      audioElement.controls = true;
      messageElement.appendChild(audioElement);
    }
  }

  // 공개 메서드
  return {
    init: init,
    sendMessage: sendMessage,
    toggleVoiceRecognition: toggleVoiceRecognition,
    toggleAutoMic: toggleAutoMic,
    login: login,
    signup: signup,
    logout: logout,
    showLoginForm: showLoginForm,
    showSignupForm: showSignupForm,
    showForgotPasswordForm: showForgotPasswordForm,
    resetPassword: resetPassword,
    showHistoryModal: showHistoryModal,
    showTodaysNews: showTodaysNews,
  };
})();

// DOM이 로드된 후 앱 초기화
document.addEventListener("DOMContentLoaded", ChatApp.init);
