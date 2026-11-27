//! Rescues a window restored wholly off-screen.
//!
//! eframe restores the saved geometry (`ui.ron`) before PhotoCraft's code runs. A position from
//! a monitor layout that no longer exists can leave the window entirely outside every display,
//! and with the custom title bar there is then no reachable title strip to drag it back
//! (observed twice on Windows 11, 2026-11-21). [`apply`] runs once at startup: a window rect
//! that touches no monitor is centred in the nearest monitor's work area, shrunk to fit it.
//! `MonitorFromWindow(DEFAULTTONEAREST)` picks the closest monitor, so a rect that misses it
//! misses them all: an intersecting monitor would be at distance zero and win instead.

pub use platform::apply;

/// New position and size for a window rect that touches no monitor: centred in `work`, shrunk
/// to fit it. `None` when the rect already intersects `monitor` (the monitor's full rect);
/// all rects are `left, top, right, bottom` in desktop coordinates.
fn rescue(rect: (i32, i32, i32, i32), monitor: (i32, i32, i32, i32), work: (i32, i32, i32, i32)) -> Option<((i32, i32), (i32, i32))> {
    let (l, t, r, b) = rect;
    let (ml, mt, mr, mb) = monitor;
    if l < mr && r > ml && t < mb && b > mt {
        return None;
    }
    let (wl, wt, wr, wb) = work;
    let (w, h) = ((r - l).min(wr - wl), (b - t).min(wb - wt));
    Some(((wl + (wr - wl - w) / 2, wt + (wb - wt - h) / 2), (w, h)))
}

#[cfg(target_os = "windows")]
mod platform {
    use eframe::wgpu::rwh::{HasWindowHandle, RawWindowHandle};
    use winsafe::{HWND, HwndPlace, POINT, SIZE, co};

    /// One startup pass; a no-op when the window touches a monitor (the usual case).
    pub fn apply(cc: &eframe::CreationContext<'_>) {
        let Some(hwnd) = find_window(cc) else { return };
        let (Ok(r), Ok(mi)) = (hwnd.GetWindowRect(), hwnd.MonitorFromWindow(co::MONITOR::DEFAULTTONEAREST).GetMonitorInfo()) else {
            return;
        };
        let rect = (r.left, r.top, r.right, r.bottom);
        let monitor = (mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom);
        let work = (mi.rcWork.left, mi.rcWork.top, mi.rcWork.right, mi.rcWork.bottom);
        let Some(((x, y), (w, h))) = super::rescue(rect, monitor, work) else { return };
        log::warn!("window restored off-screen at {rect:?}; moving it to ({x}, {y}), size {w}x{h}");
        if hwnd.IsZoomed() {
            let _ = hwnd.ShowWindow(co::SW::RESTORE);
        }
        let _ = hwnd.SetWindowPos(HwndPlace::None, POINT { x, y }, SIZE { cx: w, cy: h }, co::SWP::NOZORDER | co::SWP::NOACTIVATE);
    }

    /// The top-level window with the given raw handle, found the way `window_frame` does:
    /// `EnumWindows` hands out real `HWND`s, so no raw handle needs wrapping in `unsafe`.
    fn find_window(cc: &eframe::CreationContext<'_>) -> Option<HWND> {
        let target = match cc.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => h.hwnd.get(),
            _ => return None,
        };
        let mut found = None;
        let _ = winsafe::EnumWindows(|hwnd| {
            if hwnd.ptr() as isize == target {
                found = Some(hwnd);
                false
            } else {
                true
            }
        });
        found
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    pub fn apply(_cc: &eframe::CreationContext<'_>) {}
}

#[cfg(test)]
mod tests {
    use super::rescue;

    #[test]
    fn a_window_touching_a_monitor_is_left_alone() {
        assert_eq!(rescue((100, 100, 1540, 1000), (0, 0, 2560, 1440), (0, 0, 2560, 1392)), None);
        // Partly off the right edge still touches: the title strip is reachable to drag it back.
        assert_eq!(rescue((2000, 100, 3440, 1000), (0, 0, 2560, 1440), (0, 0, 2560, 1392)), None);
        // A second monitor to the right: touching it counts too (#4826 keeps such positions).
        assert_eq!(rescue((2560, 0, 4000, 900), (2560, 0, 4480, 1080), (2560, 0, 4480, 1040)), None);
    }

    #[test]
    fn a_window_off_every_monitor_is_centred_in_the_work_area() {
        let work = (0, 0, 2560, 1392);
        assert_eq!(rescue((-3000, -2000, -1560, -1100), (0, 0, 2560, 1440), work), Some(((560, 246), (1440, 900))));
        // Bigger than the work area: shrunk to it, which centres at its top-left.
        assert_eq!(rescue((-4000, -3000, 0, 0), (0, 0, 2560, 1440), work), Some(((0, 0), (2560, 1392))));
    }
}
