// The game's save, backed up where adb can read it, and a save brought in from another device (the Mac host, another tablet).
// The save lives in the app's internal folder (<files>/save, DM_SAVE of app/front_android.c), which adb cannot read on a release build. So at each start of the
// menu (MenuActivity), before the game starts, and again when the player comes back to the menu from the game:
//   <external>/save_backup/latest/              a copy of <files>/save, replaced each time
//   <external>/save_backup/<yyyyMMdd-HHmmss>/   the same copy kept by time; the last 5 stay
// Import: a folder <external>/save_import/ that holds files (the save's own files at its top, as in <files>/save: GameSaveData0, OptionSaveData, ...) replaces
// <files>/save at the next start of the menu. The save there is first backed up (as above, and once more in save_backup/pre-import-<time>/, which is never removed);
// then save_import becomes save_import.done-<time> and the menu shows "Save diimpor". A hibernation snapshot (<files>/hibernate/snap.bin) would bring the old game
// back and save over the imported one, so it is set aside as snap.bin.before-import. Not while the game's process is alive (it could write the save meanwhile): then
// at the next start. Files that adb pushes into Android/data are the shell's and this app cannot read them (README): first `adb shell chmod -R o+rX <folder>`;
// a folder it cannot read is left as it is and the menu says so.
// Everything is logged to logcat (tag nxsave) and to <external>/save_backup/log.txt. The external folder's files are the app's and adb reads them (as log.txt).
package com.nxnative;

import android.content.Context;
import android.util.Log;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

final class SaveBackup {
    private static final String TAG = "nxsave";
    private static final int KEEP = 5, KEEP_DAYS = 7;   // KEEP_DAYS: the first backup of each of the last 7 days stays as well
    private final Context ctx;
    private final File save, ext, backups;
    String last = "no backup line";   // the last line logged: MenuActivity hands it to the game's log (play_token)

    SaveBackup(Context c) {
        ctx = c; save = new File(c.getFilesDir(), "save");
        ext = c.getExternalFilesDir(null); backups = ext != null ? new File(ext, "save_backup") : null;
    }

    static String stamp() { return new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date()); }

    // A save waiting in save_import (what the menu checks before deciding to resume a hibernated game)
    boolean importWaiting() {
        if (ext == null) return false;
        File in = new File(ext, "save_import"); if (!in.isDirectory()) return false;
        String[] l = in.list(); return l == null || l.length > 0;   // null: there but unreadable (adb's files): run() reports it
    }

    // Backup, then the import if one waits (and allowImport). Returns the message for the player, or null. Runs off the UI thread.
    String run(boolean allowImport) {
        if (backups == null) { log("no external folder: no backup"); return null; }
        String ts = stamp();
        log("start: " + save + (save.isDirectory() ? " (" + count(save)[0] + " files)" : " (none)") + ", external " + ext);
        backup(ts);
        if (!allowImport || !importWaiting()) return null;
        File in = new File(ext, "save_import");
        if (gameAlive()) { log("save_import waits: the game's process is running (import at the next start)"); return null; }
        String bad = unreadable(in);
        if (bad != null) {
            log("save_import not imported: cannot read " + bad + " (files pushed by adb belong to the shell: adb shell chmod -R o+rX " + in + ")");
            return "Save tidak diimpor: file tidak terbaca (" + bad + "). Jalankan: adb shell chmod -R o+rX " + in;
        }
        try {
            if (save.isDirectory() && nonEmpty(save)) {
                File pre = new File(backups, "pre-import-" + ts);
                deleteTree(pre); copyTree(save, pre);
                log("before the import: the save backed up to " + pre);
            }
            File tmp = new File(ctx.getFilesDir(), "save.import.tmp"), old = new File(ctx.getFilesDir(), "save.old.tmp");
            deleteTree(tmp); deleteTree(old);
            copyTree(in, tmp);
            long[] a = count(in), b = count(tmp);
            if (a[0] != b[0] || a[1] != b[1]) throw new Exception("copy incomplete: " + b[0] + " files " + b[1] + " bytes of " + a[0] + " / " + a[1]);
            if (save.exists() && !save.renameTo(old)) throw new Exception("cannot move the old save aside");
            if (!tmp.renameTo(save)) { old.renameTo(save); throw new Exception("cannot put the imported save in place"); }
            deleteTree(old);
            File snap = new File(ctx.getFilesDir(), "hibernate/snap.bin");
            if (snap.exists()) { File aside = new File(snap.getParentFile(), "snap.bin.before-import"); aside.delete();
                log(snap.renameTo(aside) ? "hibernation snapshot set aside (" + aside.getName() + "): the game starts from the imported save" : "could not set the hibernation snapshot aside"); }
            File done = new File(ext, "save_import.done-" + ts);
            if (!in.renameTo(done)) {   // a folder adb made may not be the app's to rename: emptied instead, so it is not imported again
                deleteTree(in);
                log(in.exists() ? "WARNING: save_import could not be renamed or removed; it would be imported again: remove it with adb" : "save_import removed (it could not be renamed)");
            }
            log("imported: " + a[0] + " files, " + a[1] + " bytes from save_import into " + save + (done.exists() ? " (now " + done.getName() + ")" : ""));
            backup(ts + "-imported");   // latest/ shows what the game will load
            return "Save diimpor";
        } catch (Exception e) {
            log("import failed: " + e + " (the save in " + save + " is as before)");
            return "Save gagal diimpor: " + e.getMessage();
        }
    }

    // <files>/save into save_backup/latest/ (replaced) and save_backup/<name>/; the oldest timed copies past KEEP are removed
    private void backup(String name) {
        try {
            if (!save.isDirectory() || !nonEmpty(save)) { log("no save yet in " + save + ": nothing to back up"); return; }
            backups.mkdirs();
            File tmp = new File(backups, ".latest.tmp"), latest = new File(backups, "latest"), timed = new File(backups, name);
            deleteTree(tmp); copyTree(save, tmp);
            long[] a = count(save), b = count(tmp);
            if (a[0] != b[0] || a[1] != b[1]) throw new Exception("copy incomplete: " + b[0] + "/" + a[0] + " files");
            deleteTree(latest);
            if (!tmp.renameTo(latest)) throw new Exception("cannot rename " + tmp + " to latest");
            deleteTree(timed); copyTree(save, timed);
            String[] all = backups.list(); List<String> timedDirs = new ArrayList<>();
            if (all != null) for (String n : all) if (n.matches("\\d{8}-\\d{6}(-imported)?")) timedDirs.add(n);
            java.util.Collections.sort(timedDirs);
            // The day's first backup is kept besides the last KEEP, for the last KEEP_DAYS days (11-17: a day of test launches rotated out the only copy of a
            // player's AutoSave from that morning; every start of the menu makes a backup)
            java.util.Set<String> firstOfDay = new java.util.HashSet<>(); String day = null; List<String> days = new ArrayList<>();
            for (String n : timedDirs) { String d = n.substring(0, 8); if (!d.equals(day)) { day = d; firstOfDay.add(n); days.add(n); } }
            while (days.size() > KEEP_DAYS) firstOfDay.remove(days.remove(0));
            List<String> rot = new ArrayList<>(); for (String n : timedDirs) if (!firstOfDay.contains(n)) rot.add(n);
            for (int i = 0; i + KEEP < rot.size(); i++) deleteTree(new File(backups, rot.get(i)));
            log("backup: " + a[0] + " files, " + a[1] + " bytes -> " + latest + " and " + timed.getName());
        } catch (Exception e) { log("backup failed: " + e); }
    }

    // The game runs in the app's process ":game" (the manifest's NativeActivity)
    private boolean gameAlive() {
        try {
            android.app.ActivityManager am = (android.app.ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE);
            List<android.app.ActivityManager.RunningAppProcessInfo> ps = am.getRunningAppProcesses();
            if (ps != null) for (android.app.ActivityManager.RunningAppProcessInfo p : ps) if (p.processName.endsWith(":game")) return true;
        } catch (Exception e) { /* unknown: taken as not running */ }
        return false;
    }

    private static boolean nonEmpty(File d) { String[] l = d.list(); return l != null && l.length > 0; }
    // The first entry of the folder this app cannot read (relative name), or null
    private static String unreadable(File d) {
        String[] l = d.list(); if (l == null) return d.getName() + "/";
        for (String n : l) {
            File f = new File(d, n);
            if (f.isDirectory()) { String s = unreadable(f); if (s != null) return d.getName() + "/" + s; }
            else try (InputStream in = new FileInputStream(f)) { in.read(); } catch (Exception e) { return d.getName() + "/" + n; }
        }
        return null;
    }
    private static void copyTree(File from, File to) throws Exception {
        if (from.isDirectory()) {
            if (!to.isDirectory() && !to.mkdirs()) throw new Exception("cannot make " + to);
            String[] l = from.list(); if (l == null) throw new Exception("cannot list " + from);
            Arrays.sort(l);
            for (String n : l) copyTree(new File(from, n), new File(to, n));
        } else {
            byte[] buf = new byte[65536]; int n;
            try (InputStream in = new FileInputStream(from); OutputStream out = new FileOutputStream(to)) { while ((n = in.read(buf)) > 0) out.write(buf, 0, n); out.flush(); }
            to.setLastModified(from.lastModified());
        }
    }
    private static long[] count(File d) {   // {files, bytes}
        long[] r = {0, 0};
        if (d.isDirectory()) { String[] l = d.list(); if (l != null) for (String n : l) { long[] s = count(new File(d, n)); r[0] += s[0]; r[1] += s[1]; } }
        else if (d.isFile()) { r[0] = 1; r[1] = d.length(); }
        return r;
    }
    static void deleteTree(File f) {
        if (f.isDirectory()) { String[] l = f.list(); if (l != null) for (String n : l) deleteTree(new File(f, n)); }
        f.delete();
    }
    // Warning level: HyperOS's logcat dropped this app's info lines (no nxsave line on the Xiaomi Pad 8 while Harvest Moon was started several times)
    private void log(String s) {
        Log.w(TAG, s); last = s;
        if (backups == null) return;
        if (!backups.isDirectory() && !backups.mkdirs()) {   // the folder cannot be made (the external folder made by another app or adb): said where adb looks first
            Log.e(TAG, "cannot make " + backups + "; " + s);
            try (OutputStream o = new FileOutputStream(new File(ext, "save_backup_error.txt"), true)) { o.write(("cannot make " + backups + ": " + s + "\n").getBytes(StandardCharsets.UTF_8)); } catch (Exception e) { /* logcat has it */ }
            return;
        }
        String line = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT).format(new Date()) + " " + s + "\n";
        try (OutputStream o = new FileOutputStream(new File(backups, "log.txt"), true)) { o.write(line.getBytes(StandardCharsets.UTF_8)); } catch (Exception e) { /* logcat has it */ }
    }
}
