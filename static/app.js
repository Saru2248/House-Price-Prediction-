/* ── app.js ── House Price Prediction Dashboard ─────────────── */
'use strict';

/* ── Navbar scroll effect ───────────────────────────────────── */
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 40);
  updateActiveNav();
});

function updateActiveNav() {
  const sections = ['overview', 'predict', 'models', 'eda', 'dataset'];
  const links    = document.querySelectorAll('.nav-link');
  let current    = 'overview';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 120) current = id;
  });
  links.forEach(l => {
    l.classList.toggle('active', l.getAttribute('href') === '#' + current);
  });
}

/* ── Segmented button groups ────────────────────────────────── */
function initSegGroup(groupId, hiddenId) {
  document.getElementById(groupId).querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById(groupId).querySelectorAll('.seg-btn')
              .forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(hiddenId).value = btn.dataset.val;
    });
  });
}
initSegGroup('bed-group',  'bedrooms');
initSegGroup('bath-group', 'bathrooms');
initSegGroup('bal-group',  'balconies');
initSegGroup('park-group', 'parking');

/* ── Slider sync ────────────────────────────────────────────── */
function syncSlider(sliderId, displayId, hiddenId) {
  const slider  = document.getElementById(sliderId);
  const display = displayId ? document.getElementById(displayId) : null;
  const hidden  = hiddenId  ? document.getElementById(hiddenId)  : null;

  const update = () => {
    const v = parseFloat(slider.value);
    if (display) display.textContent = v;
    if (hidden)  hidden.value = v;
    const pct = ((v - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background =
      `linear-gradient(to right, var(--violet) ${pct}%, var(--border) ${pct}%)`;
  };
  slider.addEventListener('input', update);
  update();
}

// area slider ↔ text input
const areaSlider = document.getElementById('area_slider');
const areaInput  = document.getElementById('area_sqft');
areaSlider.addEventListener('input', () => {
  areaInput.value = areaSlider.value;
  const pct = ((areaSlider.value - areaSlider.min)/(areaSlider.max - areaSlider.min))*100;
  areaSlider.style.background =
    `linear-gradient(to right, var(--violet) ${pct}%, var(--border) ${pct}%)`;
});
areaInput.addEventListener('input', () => { areaSlider.value = areaInput.value; });

syncSlider('age_years',        'age_val',   'age_hidden');
syncSlider('floor',            'floor_val', 'floor_hidden');
syncSlider('total_floors',     'tf_val',    'tf_hidden');
syncSlider('distance_city_km', 'dist_val',  'dist_hidden');

/* ── Fetch helpers ──────────────────────────────────────────── */
async function apiFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

/* ── Load stats ─────────────────────────────────────────────── */
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');

    // Hero numbers
    animateNum('hstat-records', 0, d.total_records, 0, 1000);
    animateNum('hstat-mape',    0, 2.73, 2, 1200, '%');

    // Stat cards
    const cards = [
      { icon: '🏠', val: d.total_records.toLocaleString(), label: 'Total Properties' },
      { icon: '💰', val: `₹${d.avg_price}L`,               label: 'Avg Price' },
      { icon: '📐', val: `${d.avg_area} sq.ft`,            label: 'Avg Area' },
      { icon: '📍', val: `${d.locations} Zones`,           label: 'Locations' },
    ];
    const grid = document.getElementById('statsCards');
    grid.innerHTML = cards.map(c => `
      <div class="stat-card">
        <div class="stat-card-icon">${c.icon}</div>
        <div class="stat-card-val">${c.val}</div>
        <div class="stat-card-label">${c.label}</div>
      </div>
    `).join('');

    // Quick stats sidebar
    document.getElementById('qs-avg').textContent  = `₹${d.avg_price}L`;
    document.getElementById('qs-min').textContent  = `₹${d.min_price}L`;
    document.getElementById('qs-max').textContent  = `₹${d.max_price}L`;
    document.getElementById('qs-area').textContent = `${d.avg_area} sq.ft`;

    // Price buckets mini chart
    renderMiniBarChart('priceBuckets', d.price_distribution.labels,
      d.price_distribution.values, ['#7c3aed','#8b5cf6','#06b6d4','#10b981','#f59e0b']);

    // Location avg price chart
    const locLabels = Object.keys(d.location_avg_price);
    const locVals   = Object.values(d.location_avg_price);
    renderMiniBarChart('locationChart', locLabels, locVals,
      Array(locLabels.length).fill('#8b5cf6'));

    // Furnishing donut
    renderDonut('furnishChart', d.furnishing_counts,
      ['#7c3aed','#06b6d4','#f59e0b']);

  } catch (e) { console.error('Stats error:', e); }
}

/* ── Load model results ─────────────────────────────────────── */
async function loadModelResults() {
  try {
    const d = await apiFetch('/api/model-results');
    const colors = ['#f59e0b','#8b5cf6','#06b6d4','#94a3b8'];

    // Model cards
    const grid = document.getElementById('modelsGrid');
    grid.innerHTML = d.models.map((m, i) => `
      <div class="model-card ${i === 0 ? 'best' : ''}">
        <div class="model-name">${m.name}</div>
        <div class="model-r2" style="color:${colors[i]}">${m.r2_pct}%</div>
        <div class="model-r2-label">R² Score</div>
        <div class="model-mini">
          <span>MAE</span><span class="model-mini-val">₹${m.mae}L</span>
        </div>
        <div class="model-mini">
          <span>RMSE</span><span class="model-mini-val">₹${m.rmse}L</span>
        </div>
      </div>
    `).join('');

    // Eval strip
    const e = d.best_eval;
    document.getElementById('es-mae').textContent  = `₹${e.mae}L`;
    document.getElementById('es-rmse').textContent = `₹${e.rmse}L`;
    document.getElementById('es-r2').textContent   = `${(e.r2*100).toFixed(2)}%`;
    document.getElementById('es-mape').textContent = `${e.mape}%`;
    document.getElementById('es-cv').textContent   = `${(e.cv_mean_r2*100).toFixed(2)}%`;

    // Hero R² badge
    document.getElementById('hstat-r2').textContent =
      `${d.models[0].r2_pct}%`;

    // R² bar chart
    const chartEl = document.getElementById('r2BarChart');
    chartEl.innerHTML = d.models.map((m, i) => `
      <div class="bar-row">
        <div class="bar-label">${m.name}</div>
        <div class="bar-track">
          <div class="bar-fill" style="background:${colors[i]}" data-target="${m.r2_pct}">
            ${m.r2_pct}%
          </div>
        </div>
      </div>
    `).join('');

    // Animate bars after render
    setTimeout(() => {
      document.querySelectorAll('.bar-fill').forEach(b => {
        b.style.width = b.dataset.target + '%';
      });
    }, 200);

  } catch (e) { console.error('Models error:', e); }
}

/* ── Load EDA Charts ────────────────────────────────────────── */
async function loadCharts() {
  try {
    const d = await apiFetch('/api/charts');
    const gallery = document.getElementById('chartsGallery');
    if (!d.charts.length) {
      gallery.innerHTML = '<p class="chart-loader">No charts found. Run main.py first.</p>';
      return;
    }
    const captions = {
      '01_price_distribution.png': 'Price Distribution (Raw & Log)',
      '02_correlation_heatmap.png': 'Feature Correlation Heatmap',
      '03_price_by_location.png': 'Price by Location (Box Plot)',
      '04_area_vs_price.png': 'Area vs Price (Scatter)',
      '05_actual_vs_predicted.png': 'Actual vs Predicted Prices',
      '06_residuals.png': 'Residual Analysis',
      '07_feature_importance.png': 'Feature Importance (XGBoost)',
      '08_model_comparison.png': 'Model Comparison (MAE / RMSE / R²)',
    };
    gallery.innerHTML = d.charts.map(c => `
      <div class="chart-thumb" onclick="openLightbox('${c.url}')">
        <img src="${c.url}" alt="${c.filename}" loading="lazy" />
        <div class="chart-caption">${captions[c.filename] || c.filename}</div>
      </div>
    `).join('');
  } catch (e) {
    document.getElementById('chartsGallery').innerHTML =
      '<p class="chart-loader">Could not load charts.</p>';
  }
}

/* ── Lightbox ────────────────────────────────────────────────── */
function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lbImg').src = src;
  lb.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}
document.getElementById('lbClose').addEventListener('click', closeLightbox);
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeLightbox();
});
function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

/* ── Prediction form ────────────────────────────────────────── */
document.getElementById('predictForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn      = document.getElementById('predictBtn');
  const btnText  = btn.querySelector('.btn-text');
  const btnLoad  = btn.querySelector('.btn-loader');

  btn.disabled = true;
  btnText.classList.add('hidden');
  btnLoad.classList.remove('hidden');

  const payload = {
    area_sqft:        parseFloat(document.getElementById('area_sqft').value),
    bedrooms:         parseInt(document.getElementById('bedrooms').value),
    bathrooms:        parseInt(document.getElementById('bathrooms').value),
    balconies:        parseInt(document.getElementById('balconies').value),
    location:         document.getElementById('location').value,
    age_years:        parseInt(document.getElementById('age_hidden').value),
    parking:          parseInt(document.getElementById('parking').value),
    furnishing:       document.getElementById('furnishing').value,
    floor:            parseInt(document.getElementById('floor_hidden').value),
    total_floors:     parseInt(document.getElementById('tf_hidden').value),
    distance_city_km: parseFloat(document.getElementById('dist_hidden').value),
  };

  try {
    const result = await apiFetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    document.getElementById('resultPlaceholder').classList.add('hidden');
    const content = document.getElementById('resultContent');
    content.classList.remove('hidden');

    document.getElementById('resultPrice').textContent = result.price_formatted;
    document.getElementById('resultCrore').textContent =
      result.price_crore ? `= ₹ ${result.price_crore} Crore` : '';

    // Build meta info
    document.getElementById('resultMeta').innerHTML = `
      <strong>${payload.bedrooms}BHK</strong> · ${payload.area_sqft} sq.ft<br/>
      ${payload.location} · ${payload.furnishing}<br/>
      Floor ${payload.floor}/${payload.total_floors} · Age ${payload.age_years} yrs
    `;

    // Animated progress bar (visual relative scale)
    const pct = Math.min((result.price_lakhs / 300) * 100, 100);
    setTimeout(() => {
      document.getElementById('resultBar').style.width = pct + '%';
    }, 100);

    // Scroll result into view on mobile
    if (window.innerWidth < 900) {
      document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

  } catch (err) {
    alert('Prediction failed. Make sure the server is running and models are trained.');
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.classList.remove('hidden');
    btnLoad.classList.add('hidden');
  }
});

/* ── Mini bar chart renderer ─────────────────────────────────── */
function renderMiniBarChart(containerId, labels, values, colors) {
  const el  = document.getElementById(containerId);
  const max = Math.max(...values);
  el.innerHTML = labels.map((lbl, i) => {
    const pct = ((values[i] / max) * 100).toFixed(1);
    const val = typeof values[i] === 'number' && values[i] < 1000
      ? values[i].toFixed(1) : values[i].toLocaleString();
    return `
      <div class="mini-bar-row">
        <div class="mini-bar-label">${lbl}</div>
        <div class="mini-track">
          <div class="mini-fill" style="background:${colors[i % colors.length]};width:0%"
               data-target="${pct}"></div>
        </div>
        <div class="mini-bar-val">${val}</div>
      </div>
    `;
  }).join('');
  setTimeout(() => {
    el.querySelectorAll('.mini-fill').forEach(f => { f.style.width = f.dataset.target + '%'; });
  }, 300);
}

/* ── Donut chart renderer ────────────────────────────────────── */
function renderDonut(containerId, dataObj, colors) {
  const el     = document.getElementById(containerId);
  const labels = Object.keys(dataObj);
  const values = Object.values(dataObj);
  const total  = values.reduce((a, b) => a + b, 0);
  const r = 50, cx = 70, cy = 70, ri = 30;
  let offset = 0;
  const segments = labels.map((lbl, i) => {
    const pct   = values[i] / total;
    const angle = pct * 2 * Math.PI;
    const x1    = cx + r * Math.cos(offset);
    const y1    = cy + r * Math.sin(offset);
    offset += angle;
    const x2    = cx + r * Math.cos(offset);
    const y2    = cy + r * Math.sin(offset);
    const large = pct > 0.5 ? 1 : 0;
    return `<path d="M${cx},${cy} L${x1.toFixed(2)},${y1.toFixed(2)} A${r},${r} 0 ${large},1 ${x2.toFixed(2)},${y2.toFixed(2)} Z"
      fill="${colors[i % colors.length]}" opacity="0.85"/>`;
  });
  el.innerHTML = `
    <svg class="donut-svg" width="140" height="140" viewBox="0 0 140 140">
      ${segments.join('')}
      <circle cx="${cx}" cy="${cy}" r="${ri}" fill="var(--bg3)"/>
    </svg>
    <div class="donut-legend">
      ${labels.map((lbl, i) => `
        <div class="dleg-item">
          <div class="dleg-dot" style="background:${colors[i % colors.length]}"></div>
          <span>${lbl}: <strong>${((values[i]/total)*100).toFixed(0)}%</strong></span>
        </div>
      `).join('')}
    </div>
  `;
}

/* ── Animated number counter ─────────────────────────────────── */
function animateNum(id, from, to, decimals, duration, suffix = '') {
  const el  = document.getElementById(id);
  const fps = 60;
  const steps = Math.round(duration / (1000 / fps));
  let step = 0;
  const timer = setInterval(() => {
    step++;
    const val = from + (to - from) * (step / steps);
    el.textContent = val.toFixed(decimals) + suffix;
    if (step >= steps) { clearInterval(timer); el.textContent = to.toFixed(decimals) + suffix; }
  }, 1000 / fps);
}

/* ── Init ────────────────────────────────────────────────────── */
(async function init() {
  await Promise.all([loadStats(), loadModelResults(), loadCharts()]);
})();
