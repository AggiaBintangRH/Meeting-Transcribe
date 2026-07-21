import SwiftUI

/// Central color palette — matches the app's dark design.
enum Theme {
    static let bg        = Color(red: 0.051, green: 0.051, blue: 0.055) // main area
    static let sidebar   = Color(red: 0.086, green: 0.086, blue: 0.090)
    static let card      = Color(red: 0.125, green: 0.125, blue: 0.133)
    static let chip      = Color(red: 0.145, green: 0.145, blue: 0.153)
    static let teal      = Color(red: 0.169, green: 0.780, blue: 0.702)
    static let red       = Color(red: 0.902, green: 0.290, blue: 0.290)
    static let amber     = Color(red: 0.949, green: 0.651, blue: 0.251) // trouble, but still working on it
    static let textDim   = Color.white.opacity(0.45)
    static let textFaint = Color.white.opacity(0.30)
    static let divider   = Color.white.opacity(0.06)

    // MARK: - Text
    /// Full-strength primary text (titles, active labels).
    static let textPrimary   = Color.white
    /// Slightly softened bright text (emphasized values).
    static let textBright    = Color.white.opacity(0.9)
    /// Standard body text.
    static let textBody      = Color.white.opacity(0.85)
    /// Field/section labels.
    static let textLabel     = Color.white.opacity(0.8)
    /// Secondary text (captions, inactive labels).
    static let textSecondary = Color.white.opacity(0.7)
    /// Muted text (inactive tab labels).
    static let textMuted     = Color.white.opacity(0.65)
    /// Subtle text (least emphasis before textDim).
    static let textSubtle    = Color.white.opacity(0.6)

    // MARK: - Selection & Controls
    /// Background of the currently selected tab/section pill.
    static let selectedTabBackground = Theme.teal
    /// Text on the selected tab/section pill.
    static let selectedTabText       = Color.black
    /// Border of a toggle/checkbox in its on state.
    static let toggleOnBorder        = Theme.teal.opacity(0.55)
    /// Border highlight shown on hover.
    static let hoverHighlight        = Color.white.opacity(0.18)
    /// Accent for the Remote (conferencing) capture role, so it never reads as the
    /// Office role — teal is Office everywhere in the app.
    static let remoteRole            = Theme.amber
    /// Border of a channel tile assigned to the Remote role.
    static let remoteRoleBorder      = Theme.amber.opacity(0.55)

    // MARK: - Record Button
    /// Icon color on the idle (teal) record button.
    static let recordButtonIdleText     = Color.black.opacity(0.75)
    /// Icon color on the active (recording) button.
    static let recordButtonActiveText   = Color.white
    /// Spinner tint while the record button is preparing.
    static let recordButtonSpinnerTint  = Color.black

    // MARK: - Loading Overlay
    /// Dimming scrim behind the loading dialog.
    static let overlayDim    = Color.black.opacity(0.6)
    /// Border of the loading dialog card.
    static let overlayBorder = Color.white.opacity(0.08)
    /// Drop shadow of the loading dialog card.
    static let overlayShadow = Color.black.opacity(0.5)

    // MARK: - Sidebar
    /// Background of the circular logo/settings button.
    static let logoBackground = Color.black

    // MARK: - Transcript
    /// Text color for an identified speaker's name (rename button and plain label) —
    /// also used for "SPEAKER UNKNOWN" so it reads identically to a named speaker.
    static let speakerNameText = Theme.teal
    /// The "mm:ss – mm:ss" time range shown next to a speaker's name.
    static let timeRangeText = Theme.textFaint
    /// The "overlap" tag shown when a row was spoken over another speaker.
    static let overlapTagText = Color.orange
    /// Utterance body text — same color regardless of confirmed/unconfirmed state.
    static let bodyTextConfirmed = Color.white.opacity(0.92)
    /// Background fill of a single transcript turn card (a subtle lift above the page).
    static let rowCardBackground = Color(red: 0.078, green: 0.078, blue: 0.086)
    /// Border of a single transcript turn card (brighter so it stays visible on TV screens).
    static let rowCardBorder = Color.white.opacity(0.22)
}
