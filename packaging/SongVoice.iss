#define MyAppName "SongVoice"
#ifndef MyAppVersion
#define MyAppVersion "1.0.1"
#endif
#define MyAppPublisher "SongVoice"
#define MyAppExeName "SongVoiceLauncher.exe"

[Setup]
AppId={{7C67F17E-7994-4F75-A431-7CC7D975C45A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\SongVoice
DefaultGroupName=SongVoice
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
OutputDir=output
#ifdef OfflineBuild
OutputBaseFilename=SongVoice-{#MyAppVersion}-Offline-Setup
#else
OutputBaseFilename=SongVoice-{#MyAppVersion}-Setup
#endif
; The embedded XPU runtime is multi-gigabyte. ZIP keeps tagged CI and local
; release builds bounded; integrity is supplied by Inno's checks plus the
; published SHA256SUMS, not by a slower compression setting.
Compression=zip
SolidCompression=no
WizardStyle=modern
UninstallDisplayIcon={app}\SongVoice.exe
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "hebrew"; MessagesFile: "compiler:Languages\Hebrew.isl"

[CustomMessages]
hebrew.CreateDesktopIcon=צור קיצור דרך על שולחן העבודה
hebrew.LaunchProgram=פתח את SongVoice

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "קיצורי דרך:"; Flags: unchecked

[Files]
Source: "..\dist\SongVoice\*"; DestDir: "{app}"; Excludes: "_internal\torch-*.dist-info\licenses\third_party\*"; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef OfflineBuild
Source: "build\offline\models\*"; DestDir: "{localappdata}\SongVoice\models"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{group}\SongVoice"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\SongVoice"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM SongVoice.exe /T /F"; Flags: runhidden; RunOnceId: "StopSongVoice"

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\SongVoice"
Type: filesandordirs; Name: "{app}"

[Code]
procedure InitializeWizard;
begin
  WizardForm.Caption := 'התקנת SongVoice';
  WizardForm.WelcomeLabel1.Caption := 'ברוכים הבאים להתקנת SongVoice';
  WizardForm.WelcomeLabel2.Caption := 'האשף יתקין סביבת Python משובצת ואת כל רכיבי האפליקציה. אין צורך ב-Python או CUDA מותקנים.';
  WizardForm.NextButton.Caption := 'הבא';
  WizardForm.BackButton.Caption := 'חזרה';
  WizardForm.CancelButton.Caption := 'ביטול';
end;
