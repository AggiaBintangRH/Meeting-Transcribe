import Foundation

/// Finds the removable drives a meeting PDF may be written to (owner,
/// 2026-08-10).
///
/// "Thumbdrive" here means a volume macOS reports as REMOVABLE OR EJECTABLE.
/// Both flags are needed and neither is enough alone: a USB stick is usually
/// `removable`, while many USB-C SSDs and card readers report `ejectable` only.
/// Testing one flag silently misses half the drives people actually use.
///
/// EXCLUSION IS BY THE VOLUME'S OWN FLAGS, never by name or mount path. A user's
/// drive can be called anything, and `/Volumes/…` also holds disk images, network
/// shares and Time Machine mounts — matching on the path would offer to write a
/// client's meeting transcript to a network share.
enum ThumbdriveLocator {

    /// `Equatable` only. `Identifiable` was here for a SwiftUI picker that was
    /// never built — the chooser is an `NSPopUpButton` over names — and a
    /// conformance nothing uses is a claim about how the type is consumed.
    struct Drive: Equatable {
        let url: URL
        let name: String
    }

    /// Every removable volume currently mounted, in a stable order.
    ///
    /// Sorted by NAME rather than by mount order: the owner's picker shows this
    /// list, and a list that reorders itself between two exports is one where the
    /// user clicks the wrong row out of muscle memory.
    static func connected() -> [Drive] {
        let keys: [URLResourceKey] = [.volumeIsRemovableKey, .volumeIsEjectableKey,
                                      .volumeNameKey, .volumeIsBrowsableKey]
        let mounted = FileManager.default.mountedVolumeURLs(
            includingResourceValuesForKeys: keys,
            options: [.skipHiddenVolumes]) ?? []

        return mounted.compactMap { url -> Drive? in
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { return nil }
            // Browsable excludes volumes the user cannot open in Finder — there is
            // nothing useful about writing a PDF somewhere they cannot reach it.
            guard values.volumeIsBrowsable ?? false else { return nil }
            let removable = (values.volumeIsRemovable ?? false)
                || (values.volumeIsEjectable ?? false)
            guard removable else { return nil }
            return Drive(url: url,
                         name: values.volumeName ?? url.lastPathComponent)
        }
        .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// What the export flow should do next, given what is plugged in.
    ///
    /// A PURE FUNCTION over the drive list rather than a branch inside the export
    /// routine, so the three cases can be tested without mounting anything. The
    /// export path is the one place in this app that can block the whole UI, and
    /// "what happens when none is plugged in" deserves to be provable.
    enum Destination: Equatable {
        case single(Drive)      // exactly one → no question asked
        case choose([Drive])    // more than one → the picker
        case none               // the blocking insert-a-drive alert
    }

    static func destination(for drives: [Drive]) -> Destination {
        switch drives.count {
        case 0:  return .none
        case 1:  return .single(drives[0])
        default: return .choose(drives)
        }
    }
}
