/**
 * UI Store
 *
 * Holds persistent layout state for the three-pane shell: whether the full
 * settings sidebar is expanded out of the icon rail, and whether the right
 * stats gutter is expanded. Persisted so the layout rehydrates on reload.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  // The full settings sidebar slides out of the 44px icon rail. Default closed
  // so first load is rail-only and the dialogue takes the width.
  settingsSidebarExpanded: boolean;
  // The right gutter has two widths: collapsed (~240px) and expanded (~420px).
  gutterExpanded: boolean;
  // Expanded gutter pushes the chat narrower rather than overlaying it. Kept as
  // state so we can flip to overlay later without touching call sites.
  gutterPushesChat: boolean;

  // The source title currently hovered, in either the chat citation chips or the
  // gutter's sources list. Shared so hovering one highlights the other. Transient
  // — not persisted.
  hoveredSource: string | null;

  toggleSettingsSidebar: () => void;
  setSettingsSidebarExpanded: (expanded: boolean) => void;
  toggleGutter: () => void;
  setGutterExpanded: (expanded: boolean) => void;
  setHoveredSource: (title: string | null) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      settingsSidebarExpanded: false,
      gutterExpanded: false,
      gutterPushesChat: true,
      hoveredSource: null,

      toggleSettingsSidebar: () =>
        set((state) => ({ settingsSidebarExpanded: !state.settingsSidebarExpanded })),
      setSettingsSidebarExpanded: (expanded: boolean) =>
        set({ settingsSidebarExpanded: expanded }),
      toggleGutter: () => set((state) => ({ gutterExpanded: !state.gutterExpanded })),
      setGutterExpanded: (expanded: boolean) => set({ gutterExpanded: expanded }),
      setHoveredSource: (title: string | null) => set({ hoveredSource: title }),
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        settingsSidebarExpanded: state.settingsSidebarExpanded,
        gutterExpanded: state.gutterExpanded,
        gutterPushesChat: state.gutterPushesChat,
      }),
    }
  )
);
