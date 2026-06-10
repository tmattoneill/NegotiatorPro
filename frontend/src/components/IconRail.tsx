/**
 * IconRail
 *
 * The always-present 44px dark strip on the far left. Its top control toggles
 * the full settings sidebar in and out; the rest are navigation affordances.
 * Mirrors the np-rail block in mockups-ui/dialogue-gutter.html.
 */
import { useNavigate } from 'react-router-dom';
import {
  TbLayoutSidebarLeftExpand,
  TbLayoutSidebarLeftCollapse,
  TbMessageCircle,
  TbUsers,
  TbHistory,
  TbBooks,
  TbSettings,
  TbUser,
} from 'react-icons/tb';
import { useUIStore } from '../store/uiStore';
import { useAdminStore } from '../store/adminStore';

interface IconRailProps {
  onOpenSettings: () => void;
}

export default function IconRail({ onOpenSettings }: IconRailProps) {
  const navigate = useNavigate();
  const settingsSidebarExpanded = useUIStore((s) => s.settingsSidebarExpanded);
  const toggleSettingsSidebar = useUIStore((s) => s.toggleSettingsSidebar);
  const setView = useAdminStore((s) => s.setView);

  const iconClass = 'text-[18px] cursor-pointer transition-colors';

  return (
    <div
      className="flex w-11 shrink-0 flex-col items-center gap-[18px] py-4"
      style={{ background: 'var(--color-heading)' }}
    >
      <button
        type="button"
        onClick={toggleSettingsSidebar}
        title={settingsSidebarExpanded ? 'Hide settings' : 'Show settings'}
        className={iconClass}
        style={{ color: 'var(--color-accent)', background: 'transparent', border: 'none' }}
      >
        {settingsSidebarExpanded ? (
          <TbLayoutSidebarLeftCollapse />
        ) : (
          <TbLayoutSidebarLeftExpand />
        )}
      </button>

      <button
        type="button"
        onClick={() => setView('none')}
        title="Chat"
        className={iconClass}
        style={{ color: 'var(--sidebar-fg)', background: 'transparent', border: 'none' }}
      >
        <TbMessageCircle />
      </button>

      {/* Navigation placeholders — wired to real destinations as those land */}
      <span className={iconClass} style={{ color: 'var(--sidebar-muted)' }}>
        <TbUsers />
      </span>
      <span className={iconClass} style={{ color: 'var(--sidebar-muted)' }}>
        <TbHistory />
      </span>
      <span className={iconClass} style={{ color: 'var(--sidebar-muted)' }}>
        <TbBooks />
      </span>

      <span className="flex-1" />

      <button
        type="button"
        onClick={onOpenSettings}
        title="Settings"
        className={iconClass}
        style={{ color: 'var(--sidebar-muted)', background: 'transparent', border: 'none' }}
      >
        <TbSettings />
      </button>
      <button
        type="button"
        onClick={() => navigate('/profile')}
        title="My profile"
        className={iconClass}
        style={{ color: 'var(--sidebar-muted)', background: 'transparent', border: 'none' }}
      >
        <TbUser />
      </button>
    </div>
  );
}
