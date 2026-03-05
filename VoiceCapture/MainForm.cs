using System.Diagnostics;
using NAudio.Wave;

namespace VoiceCapture;

public class MainForm : Form
{
    // ── State ────────────────────────────────────────────────────────────────
    private string[] _lines = [];
    private int _current;
    private bool _isRecording;
    private bool _isPlaying;
    private WaveInEvent? _waveIn;
    private WaveFileWriter? _waveWriter;
    private string _recDir = "";
    private float _peakLevel;
    private float _smoothedLevel;
    private DateTime _recordStart;

    // ── Record tab controls ──────────────────────────────────────────────────
    private Label _sentenceLabel = null!;
    private Label _countLabel = null!;
    private ProgressBar _progress = null!;
    private Button _btnPrev = null!, _btnRecord = null!, _btnPlay = null!, _btnAccept = null!;
    private Panel _levelFill = null!, _levelContainer = null!;
    private Label _statusLabel = null!;
    private System.Windows.Forms.Timer _levelTimer = null!;

    // ── Generate tab controls ────────────────────────────────────────────────
    private Button _btnGenerate = null!;
    private RichTextBox _logBox = null!;
    private Label _genStatus = null!;

    // ── Colours ──────────────────────────────────────────────────────────────
    static readonly Color BgDark   = Color.FromArgb(22, 22, 26);
    static readonly Color BgMid    = Color.FromArgb(30, 30, 36);
    static readonly Color BgPanel  = Color.FromArgb(28, 28, 34);
    static readonly Color FgMain   = Color.FromArgb(230, 230, 238);
    static readonly Color FgMuted  = Color.FromArgb(130, 130, 145);
    static readonly Color Green    = Color.FromArgb(72, 200, 110);
    static readonly Color Red      = Color.FromArgb(190, 45, 45);
    static readonly Color Blue     = Color.FromArgb(45, 105, 190);
    static readonly Color Purple   = Color.FromArgb(115, 55, 185);
    static readonly Color Amber    = Color.FromArgb(220, 175, 40);

    // ════════════════════════════════════════════════════════════════════════
    public MainForm()
    {
        SuspendLayout();
        Text = "VoiceClone Recorder";
        Size = new Size(980, 680);
        MinimumSize = new Size(800, 560);
        BackColor = BgDark;
        ForeColor = FgMain;
        Font = new Font("Segoe UI", 10f);
        KeyPreview = true;
        KeyDown += OnGlobalKey;

        var tabs = new TabControl { Dock = DockStyle.Fill };
        tabs.TabPages.Add(BuildRecordTab());
        tabs.TabPages.Add(BuildGenerateTab());
        Controls.Add(tabs);

        _levelTimer = new System.Windows.Forms.Timer { Interval = 30 };
        _levelTimer.Tick += (_, _) => AnimateLevel();

        ResumeLayout();
        LoadScript();
    }

    // ── Build Record tab ─────────────────────────────────────────────────────
    private TabPage BuildRecordTab()
    {
        var page = new TabPage("  Record  ") { BackColor = BgDark, UseVisualStyleBackColor = false };

        // Sentence display
        _sentenceLabel = new Label
        {
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleCenter,
            Font = new Font("Segoe UI", 22f),
            ForeColor = Color.FromArgb(248, 248, 255),
            BackColor = BgMid,
            Padding = new Padding(36),
        };
        var centerPanel = new Panel { Dock = DockStyle.Fill, Padding = new Padding(16, 16, 16, 8) };
        centerPanel.Controls.Add(_sentenceLabel);

        // Level meter
        _levelContainer = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 10,
            BackColor = Color.FromArgb(38, 38, 46),
        };
        _levelFill = new Panel
        {
            Location = Point.Empty,
            Height = 10,
            Width = 0,
            BackColor = Green,
        };
        _levelContainer.Controls.Add(_levelFill);

        // Info bar: line counter + progress
        var infoBar = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 34,
            BackColor = BgPanel,
            Padding = new Padding(16, 6, 16, 4),
        };
        _countLabel = new Label
        {
            AutoSize = true,
            Location = new Point(16, 9),
            ForeColor = FgMuted,
            Font = new Font("Segoe UI", 9f),
            BackColor = BgPanel,
        };
        _progress = new ProgressBar
        {
            Location = new Point(175, 11),
            Height = 12,
            Width = 700,
            Style = ProgressBarStyle.Continuous,
            Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top,
        };
        infoBar.Controls.AddRange(new Control[] { _countLabel, _progress });
        infoBar.Resize += (_, _) => _progress.Width = Math.Max(100, infoBar.Width - 200);

        // Buttons
        _btnPrev   = Btn("← Prev",          BgPanel, 120);
        _btnRecord = Btn("● Record",         Red,     140);
        _btnPlay   = Btn("▶ Play",           Blue,    120);
        _btnAccept = Btn("✓ Accept & Next",  Green,   180);

        _btnPrev.Click   += (_, _) => Navigate(-1);
        _btnRecord.Click += OnRecordToggle;
        _btnPlay.Click   += OnPlayClick;
        _btnAccept.Click += OnAcceptClick;

        var btnFlow = new FlowLayoutPanel
        {
            Dock = DockStyle.Bottom,
            Height = 56,
            Padding = new Padding(8, 8, 8, 8),
            BackColor = BgPanel,
            FlowDirection = FlowDirection.LeftToRight,
        };
        btnFlow.Controls.AddRange(new Control[] { _btnPrev, _btnRecord, _btnPlay, _btnAccept });

        _statusLabel = new Label
        {
            Dock = DockStyle.Bottom,
            Height = 24,
            TextAlign = ContentAlignment.MiddleLeft,
            Padding = new Padding(16, 0, 0, 0),
            ForeColor = FgMuted,
            BackColor = BgDark,
            Font = new Font("Segoe UI", 9f),
        };

        page.Controls.Add(centerPanel);
        page.Controls.Add(_levelContainer);
        page.Controls.Add(btnFlow);
        page.Controls.Add(infoBar);
        page.Controls.Add(_statusLabel);
        return page;
    }

    // ── Build Generate tab ───────────────────────────────────────────────────
    private TabPage BuildGenerateTab()
    {
        var page = new TabPage("  Generate Voice  ") { BackColor = BgDark, UseVisualStyleBackColor = false };

        var topBar = new Panel
        {
            Dock = DockStyle.Top,
            Height = 60,
            BackColor = BgPanel,
            Padding = new Padding(16, 12, 16, 8),
        };
        _btnGenerate = Btn("⚙  Generate jenny.npy", Purple, 210);
        _btnGenerate.Location = new Point(16, 10);
        _genStatus = new Label
        {
            AutoSize = true,
            Location = new Point(240, 22),
            ForeColor = FgMuted,
            BackColor = BgPanel,
            Font = new Font("Segoe UI", 9f),
        };
        topBar.Controls.AddRange(new Control[] { _btnGenerate, _genStatus });

        _logBox = new RichTextBox
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(14, 14, 18),
            ForeColor = Color.FromArgb(170, 255, 170),
            Font = new Font("Cascadia Mono", 10f),
            ReadOnly = true,
            BorderStyle = BorderStyle.None,
            ScrollBars = RichTextBoxScrollBars.Vertical,
        };

        _btnGenerate.Click += OnGenerateClick;
        page.Controls.Add(_logBox);
        page.Controls.Add(topBar);
        return page;
    }

    private static Button Btn(string text, Color bg, int width = 140) => new()
    {
        Text = text,
        BackColor = bg,
        ForeColor = Color.White,
        FlatStyle = FlatStyle.Flat,
        Height = 40,
        Width = width,
        Font = new Font("Segoe UI", 10f),
        Margin = new Padding(4),
        FlatAppearance = { BorderSize = 0 },
        Cursor = Cursors.Hand,
    };

    // ════════════════════════════════════════════════════════════════════════
    // Script loading
    // ════════════════════════════════════════════════════════════════════════
    private void LoadScript()
    {
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var candidates = new[]
        {
            Path.Combine(baseDir, "script.txt"),
            Path.Combine(baseDir, "..", "..", "..", "script.txt"),   // project dir when running from bin/
            "script.txt",
        };

        var scriptPath = candidates.Select(Path.GetFullPath).FirstOrDefault(File.Exists);
        if (scriptPath == null)
        {
            MessageBox.Show("script.txt not found.\nExpected next to the exe or project file.",
                "Missing Script", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            _lines = ["(no script loaded)"];
        }
        else
        {
            _lines = File.ReadAllLines(scriptPath)
                         .Select(l => l.Trim())
                         .Where(l => l.Length > 0 && !l.StartsWith('#'))
                         .ToArray();
        }

        _recDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "recordings");
        Directory.CreateDirectory(_recDir);

        // Resume at first unrecorded line
        _current = _lines.Length - 1;
        for (int i = 0; i < _lines.Length; i++)
        {
            if (!File.Exists(RecPath(i))) { _current = i; break; }
        }

        RefreshUI();
    }

    private string RecPath(int i) => Path.Combine(_recDir, $"line_{i + 1:D3}.wav");

    // ════════════════════════════════════════════════════════════════════════
    // UI refresh
    // ════════════════════════════════════════════════════════════════════════
    private void RefreshUI()
    {
        if (_lines.Length == 0) return;

        _sentenceLabel.Text = _lines[_current];
        _countLabel.Text = $"Line {_current + 1} of {_lines.Length}";
        _progress.Value = Math.Clamp((int)((double)(_current + 1) / _lines.Length * 100), 0, 100);

        bool hasRec = File.Exists(RecPath(_current));
        _btnPlay.Enabled   = hasRec && !_isRecording && !_isPlaying;
        _btnAccept.Enabled = hasRec && !_isRecording && !_isPlaying;
        _btnPrev.Enabled   = _current > 0 && !_isRecording && !_isPlaying;
        _btnRecord.Enabled = !_isPlaying;

        if (!_isRecording && !_isPlaying)
            SetStatus(hasRec
                ? "Ready — line recorded. Re-record or press Enter to accept."
                : "Ready to record. Press Space to start.");
    }

    private void SetStatus(string msg) => _statusLabel.Text = $"  {msg}";

    // ════════════════════════════════════════════════════════════════════════
    // Recording
    // ════════════════════════════════════════════════════════════════════════
    private void OnRecordToggle(object? s, EventArgs e)
    {
        if (_isRecording) StopRecording();
        else StartRecording();
    }

    private void StartRecording()
    {
        try
        {
            _waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(24000, 16, 1),
                BufferMilliseconds = 50,
            };
            _waveWriter = new WaveFileWriter(RecPath(_current), _waveIn.WaveFormat);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Could not open microphone:\n{ex.Message}",
                "Microphone Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        _waveIn.DataAvailable += (_, e) =>
        {
            _waveWriter?.Write(e.Buffer, 0, e.BytesRecorded);

            float peak = 0;
            for (int i = 0; i + 1 < e.BytesRecorded; i += 2)
                peak = Math.Max(peak, Math.Abs(BitConverter.ToInt16(e.Buffer, i) / 32768f));
            _peakLevel = peak;
        };

        _waveIn.RecordingStopped += (_, e) =>
        {
            _waveWriter?.Flush();
            _waveWriter?.Dispose();
            _waveWriter = null;

            if (e.Exception != null)
                BeginInvoke(() => MessageBox.Show($"Recording error: {e.Exception.Message}"));

            BeginInvoke(() =>
            {
                _isRecording = false;
                _levelTimer.Stop();
                _peakLevel = _smoothedLevel = 0;
                AnimateLevel();
                _btnRecord.Text = "● Record";
                _btnRecord.BackColor = Red;
                var elapsed = (DateTime.Now - _recordStart).TotalSeconds;
                SetStatus($"Recorded {elapsed:F1}s — play back or accept.");
                RefreshUI();
            });
        };

        _waveIn.StartRecording();
        _isRecording = true;
        _recordStart = DateTime.Now;
        _btnRecord.Text = "■ Stop";
        _btnRecord.BackColor = Amber;
        _btnPrev.Enabled = _btnPlay.Enabled = _btnAccept.Enabled = false;
        SetStatus("Recording… (Space or click Stop when done)");
        _levelTimer.Start();
    }

    private void StopRecording()
    {
        _waveIn?.StopRecording();
        _waveIn?.Dispose();
        _waveIn = null;
    }

    private void AnimateLevel()
    {
        // Exponential smoothing for visual decay
        _smoothedLevel = _isRecording
            ? _smoothedLevel * 0.55f + _peakLevel * 0.45f
            : _smoothedLevel * 0.7f;

        int w = (int)(_smoothedLevel * _levelContainer.Width);
        _levelFill.Width = Math.Max(0, Math.Min(w, _levelContainer.Width));

        _levelFill.BackColor = _smoothedLevel > 0.88f ? Red
                             : _smoothedLevel > 0.65f ? Amber
                             : Green;
    }

    // ════════════════════════════════════════════════════════════════════════
    // Playback
    // ════════════════════════════════════════════════════════════════════════
    private void OnPlayClick(object? s, EventArgs e)
    {
        var path = RecPath(_current);
        if (!File.Exists(path)) return;

        _isPlaying = true;
        _btnPlay.Enabled = _btnAccept.Enabled = _btnPrev.Enabled = _btnRecord.Enabled = false;
        SetStatus("Playing back…");

        Task.Run(() =>
        {
            try
            {
                using var reader = new AudioFileReader(path);
                using var output = new WaveOutEvent();
                output.Init(reader);
                output.Play();
                while (output.PlaybackState == PlaybackState.Playing)
                    Thread.Sleep(40);
            }
            catch (Exception ex)
            {
                BeginInvoke(() => MessageBox.Show($"Playback error: {ex.Message}"));
            }
            finally
            {
                BeginInvoke(() => { _isPlaying = false; SetStatus("Playback done."); RefreshUI(); });
            }
        });
    }

    // ════════════════════════════════════════════════════════════════════════
    // Navigation
    // ════════════════════════════════════════════════════════════════════════
    private void Navigate(int delta)
    {
        var next = Math.Clamp(_current + delta, 0, _lines.Length - 1);
        if (next != _current) { _current = next; RefreshUI(); }
    }

    private void OnAcceptClick(object? s, EventArgs e)
    {
        if (_current < _lines.Length - 1)
            Navigate(1);
        else
            SetStatus("All lines recorded! Switch to the 'Generate Voice' tab.");
    }

    private void OnGlobalKey(object? s, KeyEventArgs e)
    {
        if (_isRecording && e.KeyCode == Keys.Space)
        {
            StopRecording();
            e.SuppressKeyPress = true;
        }
        else if (!_isRecording && !_isPlaying)
        {
            switch (e.KeyCode)
            {
                case Keys.Space:
                    StartRecording();
                    e.SuppressKeyPress = true;
                    break;
                case Keys.Enter:
                    if (_btnAccept.Enabled) OnAcceptClick(null, EventArgs.Empty);
                    e.SuppressKeyPress = true;
                    break;
                case Keys.Left:
                    if (_btnPrev.Enabled) Navigate(-1);
                    break;
                case Keys.P:
                    if (_btnPlay.Enabled) OnPlayClick(null, EventArgs.Empty);
                    break;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Generate Voice
    // ════════════════════════════════════════════════════════════════════════
    private async void OnGenerateClick(object? s, EventArgs e)
    {
        var embedDir = FindDir(AppDomain.CurrentDomain.BaseDirectory,
            ["embed", "../embed", "../../embed", "../../../embed", "../../../../embed"]);

        if (embedDir == null)
        {
            AppendLog("ERROR: Could not find the embed/ directory.\n");
            AppendLog("Make sure you have the embed/ folder from the repo.\n");
            return;
        }

        _btnGenerate.Enabled = false;
        _logBox.Clear();
        _genStatus.Text = "Running…";
        AppendLog($"Recordings: {_recDir}\n");
        AppendLog($"Embed dir:  {embedDir}\n\n");

        await Task.Run(() => RunPy(embedDir, "preprocess.py", $"--input \"{_recDir}\""));
        await Task.Run(() => RunPy(embedDir, "extract_embedding.py", ""));

        _btnGenerate.Enabled = true;
        _genStatus.Text = "Done — check output above.";
    }

    private void RunPy(string dir, string script, string args)
    {
        AppendLog($"=== {script} ===\n");
        var psi = new ProcessStartInfo("python", $"{script} {args}")
        {
            WorkingDirectory = dir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        try
        {
            using var proc = Process.Start(psi)!;
            proc.OutputDataReceived += (_, e) => { if (e.Data != null) AppendLog(e.Data + "\n"); };
            proc.ErrorDataReceived  += (_, e) => { if (e.Data != null) AppendLog("[!] " + e.Data + "\n"); };
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            proc.WaitForExit();
            AppendLog($"Exit code: {proc.ExitCode}\n\n");
        }
        catch (Exception ex)
        {
            AppendLog($"Failed to start python: {ex.Message}\n");
            AppendLog("Ensure Python is on PATH and embed/venv is activated, or run manually.\n");
        }
    }

    private void AppendLog(string text)
    {
        if (InvokeRequired) { BeginInvoke(() => AppendLog(text)); return; }
        _logBox.AppendText(text);
        _logBox.ScrollToCaret();
    }

    private static string? FindDir(string baseDir, string[] relatives)
    {
        foreach (var rel in relatives)
        {
            var full = Path.GetFullPath(Path.Combine(baseDir, rel));
            if (Directory.Exists(full)) return full;
        }
        return null;
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        if (_isRecording) StopRecording();
        _levelTimer.Dispose();
        base.OnFormClosing(e);
    }
}
