document.addEventListener("DOMContentLoaded", function () {
  // Packages list is injected at build time from earthkit-packages.yml via earthkit-packages.js
  var packages = window.earthkitPackages || [];
  if (!packages.length) return;

  // Find the "Select earthkit documentation" caption in the sidebar
  var captions = document.querySelectorAll(".caption-text");
  var targetCaption = null;
  for (var i = 0; i < captions.length; i++) {
    if (captions[i].textContent.trim() === "Select earthkit documentation") {
      targetCaption = captions[i];
      break;
    }
  }
  if (!targetCaption) return;

  var captionP = targetCaption.closest("p.caption");
  if (!captionP) return;

  // Build the <select> from the YAML-derived packages list
  var select = document.createElement("select");
  select.setAttribute("aria-label", "Select earthkit documentation");

  packages.forEach(function (p) {
    var opt = document.createElement("option");
    opt.value = p.url;
    opt.textContent = p.name;
    if (p.default) opt.selected = true;
    select.appendChild(opt);
  });

  // Build the "Go" button
  var btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "Go";
  btn.addEventListener("click", function () {
    var url = select.value;
    if (url) window.open(url, "_blank", "noopener,noreferrer");
  });

  // Wrapper div
  var wrapper = document.createElement("div");
  wrapper.className = "ek-project-selector";
  wrapper.appendChild(select);
  wrapper.appendChild(btn);

  // Replace the toctree <ul> with the wrapper if present, otherwise insert after caption
  var ul = captionP.nextElementSibling;
  while (ul && ul.tagName !== "UL") {
    ul = ul.nextElementSibling;
  }
  if (ul) {
    ul.parentNode.replaceChild(wrapper, ul);
  } else {
    captionP.insertAdjacentElement("afterend", wrapper);
  }
});
