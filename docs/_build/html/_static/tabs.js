var labels_by_text = {};

function ready() {
  var li = document.getElementsByClassName("tab-label");
  const urlParams = new URLSearchParams(window.location.search);
  const tabs = urlParams.getAll("tabs");

  for (const label of li) {
    label.onclick = onLabelClick;
    const text = label.textContent;
    if (!labels_by_text[text]) {
      labels_by_text[text] = [];
    }
    labels_by_text[text].push(label);
  }

  for (const tab of tabs) {
    for (label of labels_by_text[tab]) {
      label.previousSibling.checked = true;
    }
  }
}

function onLabelClick() {
  // Activate other labels with the same text.
  for (label of labels_by_text[this.textContent]) {
    label.previousSibling.checked = true;
  }
}
document.addEventListener("DOMContentLoaded", ready, false);
