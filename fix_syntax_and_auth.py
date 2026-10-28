import re

file_path = 'scripts/build_frontend.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the auth bypass script at the bottom
bypass_pattern = r'<script>\s*\(function\s*\(\)\s*\{[\s\S]*?unlock\(\);\s*document\.addEventListener\("DOMContentLoaded", unlock\);[\s\S]*?<\/script>'
content, count_removed = re.subn(bypass_pattern, '', content)
print(f"Removed {count_removed} auth bypass script(s)!")

# 2. Fix the JavaScript syntax error in tabular code
# Replace SAMPLE_CSV_CONTENT and parseSqlInsertToCsv with syntax-safe versions
old_tabular_js_pattern = r'// ==========================================\s*// TABULAR INGESTION \(POST /analyze-tabular\)[\s\S]*?(?=<\/script>)'

new_tabular_js = """// ==========================================
    // TABULAR INGESTION (POST /analyze-tabular)
    // ==========================================
    function openTabularModal() {
      const modal = document.getElementById("tabular-modal");
      if (modal) modal.classList.add("open");
    }

    function closeTabularModal(e) {
      if (e && e.target !== document.getElementById("tabular-modal") && !e.target.closest(".modal-close-btn") && !e.target.closest(".btn-outline")) return;
      const modal = document.getElementById("tabular-modal");
      if (modal) modal.classList.remove("open");
    }

    const SAMPLE_CSV_CONTENT = [
      "user_id,merchant_id,device_id,shared_ip,amount,timestamp",
      "usr_101,merch_alpha,dev_901,192.168.1.10,4800.00,1698471000",
      "usr_102,merch_alpha,dev_901,192.168.1.10,4950.00,1698471300",
      "usr_103,merch_beta,dev_902,192.168.1.10,4900.00,1698471600",
      "usr_104,merch_beta,dev_902,10.0.4.15,120.00,1698472000",
      "usr_105,merch_gamma,dev_903,10.0.4.15,3500.00,1698472400",
      "usr_106,merch_gamma,dev_903,192.168.1.10,4850.00,1698472800",
      "usr_107,merch_delta,dev_904,172.16.0.8,4920.00,1698473200",
      "usr_108,merch_delta,dev_904,172.16.0.8,4880.00,1698473600"
    ].join(String.fromCharCode(10));

    function downloadSampleCsv() {
      const blob = new Blob([SAMPLE_CSV_CONTENT], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", "darknode_sample_transactions.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showToast("Downloaded sample transaction CSV table.", "Sample Table", "success");
    }

    function loadDemoTabularCsv() {
      const blob = new Blob([SAMPLE_CSV_CONTENT], { type: "text/csv" });
      selectedTabularFile = new File([blob], "darknode_sample_transactions.csv", { type: "text/csv" });

      const display = document.getElementById("tabular-filename-display");
      if (display) {
        display.innerHTML = '<span style="color:var(--yellow-primary); font-weight:700;">darknode_sample_transactions.csv</span> (Sample Table Loaded)';
      }
      showToast("Loaded sample AML transaction table into input.", "Sample Table", "info");
    }

    async function handleTabularFileSelect(event) {
      const file = event.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();
      const display = document.getElementById("tabular-filename-display");

      if (ext === 'sql') {
        try {
          const text = await file.text();
          const parsedCsv = parseSqlInsertToCsv(text);
          if (parsedCsv) {
            const blob = new Blob([parsedCsv], { type: "text/csv" });
            selectedTabularFile = new File([blob], file.name.replace(/\\.sql$/i, '.csv'), { type: "text/csv" });
            if (display) {
              display.innerHTML = '<span style="color:var(--yellow-primary); font-weight:700;">' + file.name + '</span> (Parsed SQL to Table)';
            }
            showToast("Successfully parsed SQL statements into table format.", "SQL Processed", "info");
            return;
          }
        } catch (e) {
          console.warn("SQL parse fallback:", e);
        }
      }

      selectedTabularFile = file;
      if (display) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        display.innerHTML = '<span style="color:var(--yellow-primary); font-weight:700;">' + file.name + '</span> (' + sizeMb + ' MB)';
      }
    }

    function parseSqlInsertToCsv(sqlText) {
      const rows = [];
      const lines = sqlText.split(';');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const match = line.match(/INSERT\\s+INTO\\s+[^\\(]*\\(([^\\)]+)\\)\\s*VALUES\\s*([\\s\\S]*)/i);
        if (match) {
          const cols = match[1].split(',').map(c => c.trim().replace(/[`"\\[\\]]/g, ''));
          if (rows.length === 0) rows.push(cols.join(','));
          const valsMatch = match[2].match(/\\(([^\\)]+)\\)/g);
          if (valsMatch) {
            for (let j = 0; j < valsMatch.length; j++) {
              const cleanVals = valsMatch[j].replace(/[\\(\\)]/g, '').split(',').map(v => v.trim().replace(/^['"]|['"]$/g, ''));
              rows.push(cleanVals.join(','));
            }
          }
        }
      }
      return rows.length > 1 ? rows.join(String.fromCharCode(10)) : null;
    }

    async function submitTabularIngestion() {
      if (!selectedTabularFile) {
        showToast("Please choose a CSV, Excel (.xlsx/.xls), or SQL table first.", "Table Required", "warning");
        return;
      }

      const runBtn = document.getElementById("run-tabular-btn");
      const statusBox = document.getElementById("tabular-status-box");
      const urlInput = document.getElementById("api-url");
      const url = (urlInput ? urlInput.value.trim() : "").replace(/\\/$/, "") || "http://localhost:8000";

      const threshInput = document.getElementById("tabular-threshold");
      const threshold = threshInput ? parseFloat(threshInput.value) : 0.5;

      const topKInput = document.getElementById("tabular-top-k");
      const topK = topKInput ? parseInt(topKInput.value, 10) : 10;

      const formData = new FormData();
      formData.append("file", selectedTabularFile);
      formData.append("threshold", threshold.toString());
      formData.append("top_k", topK.toString());

      const colUser = document.getElementById("col-user-id")?.value.trim();
      const colMerchant = document.getElementById("col-merchant-id")?.value.trim();
      const colDevice = document.getElementById("col-device-id")?.value.trim();
      const colIp = document.getElementById("col-shared-ip")?.value.trim();
      const colAmount = document.getElementById("col-amount")?.value.trim();
      const colTimestamp = document.getElementById("col-timestamp")?.value.trim();

      if (colUser) formData.append("user_id_column", colUser);
      if (colMerchant) formData.append("merchant_id_column", colMerchant);
      if (colDevice) formData.append("device_id_column", colDevice);
      if (colIp) formData.append("shared_ip_column", colIp);
      if (colAmount) formData.append("amount_column", colAmount);
      if (colTimestamp) formData.append("timestamp_column", colTimestamp);

      if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerHTML = `
          <svg class="spin-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
          Ingesting & Inferring...
        `;
      }

      if (statusBox) {
        statusBox.style.display = "block";
        statusBox.style.background = "rgba(250, 204, 21, 0.08)";
        statusBox.style.color = "var(--yellow-primary)";
        statusBox.style.border = "1px solid var(--yellow-border)";
        statusBox.innerText = "POST /analyze-tabular: Converting rows to Heterogeneous Graph & running MPNN...";
      }

      try {
        const res = await fetch(`${url}/analyze-tabular`, {
          method: "POST",
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        TABULAR_RINGS = data.rings || [];

        // Update Tabular Tab in left panel
        const tabTabular = document.getElementById("dataset-tab-tabular");
        const countSpan = document.getElementById("tabular-rings-count");
        if (tabTabular) tabTabular.style.display = "inline-flex";
        if (countSpan) countSpan.innerText = TABULAR_RINGS.length;

        // Update header stats
        if (data.metadata) {
          const u = document.getElementById("stat-users");
          const tx = document.getElementById("stat-txs");
          if (u && data.metadata.num_users) u.innerText = data.metadata.num_users.toLocaleString();
          if (tx && data.metadata.total_transactions) tx.innerText = data.metadata.total_transactions.toLocaleString();
        }

        if (statusBox) {
          statusBox.style.background = "rgba(34, 197, 94, 0.1)";
          statusBox.style.color = "#4ade80";
          statusBox.style.border = "1px solid rgba(34, 197, 94, 0.3)";
          statusBox.innerText = `Ingestion successful! Extracted ${TABULAR_RINGS.length} syndicates from ${data.metadata?.total_transactions || 0} transactions.`;
        }

        setTimeout(() => {
          closeTabularModal();
          switchDatasetSection('tabular');
          showToast(`Successfully analyzed table: ${TABULAR_RINGS.length} fraud syndicates detected.`, "Tabular Analysis Complete", "success");
        }, 800);

      } catch (err) {
        if (statusBox) {
          statusBox.style.display = "block";
          statusBox.style.background = "rgba(239, 68, 68, 0.1)";
          statusBox.style.color = "#f87171";
          statusBox.style.border = "1px solid rgba(239, 68, 68, 0.3)";
          statusBox.innerText = `Ingestion failed: ${err.message}`;
        }
        showToast(`Tabular ingestion error: ${err.message}`, "Ingestion Error", "error");
      } finally {
        if (runBtn) {
          runBtn.disabled = false;
          runBtn.innerHTML = `
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Analyze Data Table
          `;
        }
      }
    }
"""

match = re.search(old_tabular_js_pattern, content)
if match:
    content = content[:match.start()] + new_tabular_js.strip() + content[match.end():]
    print("Replaced tabular JavaScript with syntax-safe implementation!")
else:
    print("Warning: old tabular js pattern not found!")

# 3. Add Demo Login Quick Fill on Auth Form so users can instantly test the sign in page
demo_login_btn = """<div style="display: flex; gap: 8px; margin-top: 14px;">
              <button type="button" class="btn btn-outline" style="width: 100%; height: 32px; font-size: 11px; color: var(--yellow-primary); border-color: var(--yellow-border);" onclick="quickFillDemoAuth()">
                ⚡ Autofill Demo Credentials
              </button>
            </div>"""

if 'id="auth-submit"' in content and 'quickFillDemoAuth' not in content:
    content = content.replace(
        '<button class="auth-submit" id="auth-submit" type="submit">VERIFY EMPLOYEE ACCESS</button>',
        '<button class="auth-submit" id="auth-submit" type="submit">VERIFY EMPLOYEE ACCESS</button>' + demo_login_btn,
        1
    )
    print("Added Quick Demo Credentials button to sign-in form!")

# Add quickFillDemoAuth function to auth script
auth_quick_fn = """
    function quickFillDemoAuth() {
      const email = document.getElementById("auth-email");
      const empId = document.getElementById("auth-employee-id");
      const sector = document.getElementById("auth-sector");
      const pass = document.getElementById("auth-password");
      if (email) email.value = "investigator@darknode.gov";
      if (empId) empId.value = "DN-AML-0042";
      if (sector) sector.value = "AML";
      if (pass) pass.value = "FederalIntelligence2026!";
      setAuthMessage("Demo credentials loaded. Click VERIFY to enter.", true);
    }
"""
if 'function initAuthGate()' in content and 'quickFillDemoAuth' not in content:
    content = content.replace('function initAuthGate()', auth_quick_fn + '\n    function initAuthGate()', 1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Saved scripts/build_frontend.py successfully!")

