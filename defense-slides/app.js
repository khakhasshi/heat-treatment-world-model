(function () {
  "use strict";

  var stage = document.getElementById("stage");
  var slides = Array.from(document.querySelectorAll(".slide"));
  var progressBar = document.getElementById("progressBar");
  var slideCounter = document.getElementById("slideCounter");
  var sectionLabel = document.getElementById("sectionLabel");
  var overviewDialog = document.getElementById("overviewDialog");
  var overviewGrid = document.getElementById("overviewGrid");
  var notesPanel = document.getElementById("notesPanel");
  var notesContent = document.getElementById("notesContent");
  var currentIndex = 0;
  var touchStartX = null;

  function indexFromHash() {
    var match = window.location.hash.match(/(\d+)/);
    if (!match) return 0;
    return Math.min(Math.max(Number(match[1]) - 1, 0), slides.length - 1);
  }

  function resizeStage() {
    var scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    document.documentElement.style.setProperty("--scale", String(scale));
  }

  function updateOverviewSelection() {
    Array.from(overviewGrid.children).forEach(function (button, index) {
      button.setAttribute("aria-current", index === currentIndex ? "true" : "false");
    });
  }

  function updateNotes() {
    var note = slides[currentIndex].querySelector(".speaker-note");
    notesContent.textContent = note ? note.textContent.trim() : "本页无备注。";
  }

  function showSlide(index, updateHistory) {
    currentIndex = (index + slides.length) % slides.length;
    slides.forEach(function (slide, slideIndex) {
      var isActive = slideIndex === currentIndex;
      slide.classList.toggle("active", isActive);
      slide.setAttribute("aria-hidden", isActive ? "false" : "true");
    });

    var current = slides[currentIndex];
    var isBackup = current.dataset.kind === "backup";
    slideCounter.textContent = String(currentIndex + 1) + " / " + String(slides.length);
    sectionLabel.textContent = isBackup ? "BACKUP" : "MAIN";
    progressBar.style.width = String(((currentIndex + 1) / slides.length) * 100) + "%";
    document.title = current.dataset.title + " | 金属热处理世界模型答辩";
    updateNotes();
    updateOverviewSelection();

    if (updateHistory !== false) {
      window.history.replaceState(null, "", "#/" + String(currentIndex + 1));
    }
  }

  function toggleOverview() {
    if (overviewDialog.open) {
      overviewDialog.close();
    } else {
      overviewDialog.showModal();
      updateOverviewSelection();
    }
  }

  function toggleNotes(forceOpen) {
    var shouldOpen = typeof forceOpen === "boolean"
      ? forceOpen
      : !notesPanel.classList.contains("open");
    notesPanel.classList.toggle("open", shouldOpen);
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(function () {});
    } else {
      document.exitFullscreen().catch(function () {});
    }
  }

  slides.forEach(function (slide, index) {
    var button = document.createElement("button");
    var number = document.createElement("span");
    var title = document.createElement("span");
    button.type = "button";
    button.className = slide.dataset.kind === "backup" ? "backup-item" : "";
    button.setAttribute("aria-label", "跳转到第 " + String(index + 1) + " 页：" + slide.dataset.title);
    number.className = "overview-number";
    number.textContent = String(index + 1).padStart(2, "0") + (slide.dataset.kind === "backup" ? " · BACKUP" : "");
    title.className = "overview-title";
    title.textContent = slide.dataset.title;
    button.appendChild(number);
    button.appendChild(title);
    button.addEventListener("click", function () {
      showSlide(index);
      overviewDialog.close();
    });
    overviewGrid.appendChild(button);
  });

  document.getElementById("prevBtn").addEventListener("click", function () {
    showSlide(currentIndex - 1);
  });
  document.getElementById("nextBtn").addEventListener("click", function () {
    showSlide(currentIndex + 1);
  });
  document.getElementById("overviewBtn").addEventListener("click", toggleOverview);
  document.getElementById("closeOverviewBtn").addEventListener("click", function () {
    overviewDialog.close();
  });
  document.getElementById("notesBtn").addEventListener("click", function () {
    toggleNotes();
  });
  document.getElementById("closeNotesBtn").addEventListener("click", function () {
    toggleNotes(false);
  });
  document.getElementById("fullscreenBtn").addEventListener("click", toggleFullscreen);

  document.addEventListener("keydown", function (event) {
    if (overviewDialog.open && event.key !== "Escape") return;
    if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") {
      event.preventDefault();
      showSlide(currentIndex + 1);
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      showSlide(currentIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      showSlide(0);
    } else if (event.key === "End") {
      event.preventDefault();
      showSlide(slides.length - 1);
    } else if (event.key.toLowerCase() === "o") {
      event.preventDefault();
      toggleOverview();
    } else if (event.key.toLowerCase() === "n") {
      event.preventDefault();
      toggleNotes();
    } else if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      toggleFullscreen();
    } else if (event.key === "Escape") {
      toggleNotes(false);
    }
  });

  document.addEventListener("touchstart", function (event) {
    touchStartX = event.changedTouches[0].clientX;
  }, { passive: true });

  document.addEventListener("touchend", function (event) {
    if (touchStartX === null) return;
    var delta = event.changedTouches[0].clientX - touchStartX;
    touchStartX = null;
    if (Math.abs(delta) < 50) return;
    showSlide(delta < 0 ? currentIndex + 1 : currentIndex - 1);
  }, { passive: true });

  window.addEventListener("hashchange", function () {
    showSlide(indexFromHash(), false);
  });
  window.addEventListener("resize", resizeStage);

  resizeStage();
  showSlide(indexFromHash(), false);
})();
