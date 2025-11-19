/**
 * SettingsPanel Component
 *
 * Container for all user settings including model selection and feature toggles.
 */

import ModelSelector from './ModelSelector';
import { useSettingsStore } from '../store/settingsStore';

export default function SettingsPanel() {
  const { usePremiumModel, usePreprocessing, setPremiumModel, setPreprocessing } =
    useSettingsStore();

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h3>Settings</h3>
      </div>

      {/* Model Selection */}
      <ModelSelector />

      {/* Feature Toggles */}
      <div className="settings-toggles">
        <label className="toggle-item">
          <input
            type="checkbox"
            checked={usePremiumModel}
            onChange={(e) => setPremiumModel(e.target.checked)}
          />
          <span className="toggle-label">
            Use Premium Model
            <span className="toggle-hint">Override with configured premium model</span>
          </span>
        </label>

        <label className="toggle-item">
          <input
            type="checkbox"
            checked={usePreprocessing}
            onChange={(e) => setPreprocessing(e.target.checked)}
          />
          <span className="toggle-label">
            Optimize Text
            <span className="toggle-hint">Remove boilerplate to reduce tokens</span>
          </span>
        </label>
      </div>
    </div>
  );
}
