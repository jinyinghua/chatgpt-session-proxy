"""
OpenAI Sentinel PoW (Proof of Work) solver.
Adapted from chatgpt-image-studio (Go) — handler/pow.go
Uses SHA3-512 (Python hashlib built-in).
"""
import hashlib
import uuid
import random
import base64
import json
import time
from datetime import datetime, timezone, timedelta

# ─── Constants from Go code ────────────────────────────────────────────

SCREEN_SIZES = [3000, 4000, 3120, 4160]
CORE_COUNTS = [8, 16, 24, 32]

NAVIGATOR_KEY = [
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "share−function share() { [native code] }",
    "webdriver−false",
    "managed−[object NavigatorManagedData]",
    "canShare−function canShare() { [native code] }",
    "vendor−Google Inc.",
    "vendor−Google Inc.",
    "mediaDevices−[object MediaDevices]",
    "vibrate−function vibrate() { [native code] }",
    "storageBuckets−[object StorageBucketManager]",
    "mediaCapabilities−[object MediaCapabilities]",
    "getGamepads−function getGamepads() { [native code] }",
    "bluetooth−[object Bluetooth]",
    "share−function share() { [native code] }",
    "cookieEnabled−true",
    "virtualKeyboard−[object VirtualKeyboard]",
    "product−Gecko",
    "mediaDevices−[object MediaDevices]",
    "canShare−function canShare() { [native code] }",
    "getGamepads−function getGamepads() { [native code] }",
    "product−Gecko",
    "xr−[object XRSystem]",
    "clipboard−[object Clipboard]",
    "storageBuckets−[object StorageBucketManager]",
    "unregisterProtocolHandler−function unregisterProtocolHandler() { [native code] }",
    "productSub−20030107",
    "login−[object NavigatorLogin]",
    "vendorSub−",
    "login−[object NavigatorLogin]",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "mediaDevices−[object MediaDevices]",
    "locks−[object LockManager]",
    "webkitGetUserMedia−function webkitGetUserMedia() { [native code] }",
    "vendor−Google Inc.",
    "xr−[object XRSystem]",
    "mediaDevices−[object MediaDevices]",
    "virtualKeyboard−[object VirtualKeyboard]",
    "virtualKeyboard−[object VirtualKeyboard]",
    "appName−Netscape",
    "storageBuckets−[object StorageBucketManager]",
    "presentation−[object Presentation]",
    "onLine−true",
    "mimeTypes−[object MimeTypeArray]",
    "credentials−[object CredentialsContainer]",
    "presentation−[object Presentation]",
    "getGamepads−function getGamepads() { [native code] }",
    "vendorSub−",
    "virtualKeyboard−[object VirtualKeyboard]",
    "serviceWorker−[object ServiceWorkerContainer]",
    "xr−[object XRSystem]",
    "product−Gecko",
    "keyboard−[object Keyboard]",
    "gpu−[object GPU]",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "doNotTrack",
    "clearAppBadge−function clearAppBadge() { [native code] }",
    "presentation−[object Presentation]",
    "serial−[object Serial]",
    "locks−[object LockManager]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "locks−[object LockManager]",
    "requestMediaKeySystemAccess−function requestMediaKeySystemAccess() { [native code] }",
    "vendor−Google Inc.",
    "pdfViewerEnabled−true",
    "language−zh-CN",
    "setAppBadge−function setAppBadge() { [native code] }",
    "geolocation−[object Geolocation]",
    "userAgentData−[object NavigatorUAData]",
    "mediaCapabilities−[object MediaCapabilities]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "getUserMedia−function getUserMedia() { [native code] }",
    "mediaDevices−[object MediaDevices]",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "sendBeacon−function sendBeacon() { [native code] }",
    "hardwareConcurrency−32",
    "credentials−[object CredentialsContainer]",
    "storage−[object StorageManager]",
    "cookieEnabled−true",
    "pdfViewerEnabled−true",
    "windowControlsOverlay−[object WindowControlsOverlay]",
    "scheduling−[object Scheduling]",
    "pdfViewerEnabled−true",
    "hardwareConcurrency−32",
    "xr−[object XRSystem]",
    "webdriver−false",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "bluetooth−[object Bluetooth]",
]

DOCUMENT_KEY = [
    "_reactListeningo743lnnpvdg",
    "location",
]

WINDOW_KEY = [
    "0", "window", "self", "document", "name", "location", "customElements",
    "history", "navigation", "locationbar", "menubar", "personalbar",
    "scrollbars", "statusbar", "toolbar", "status", "closed", "frames",
    "length", "top", "opener", "parent", "frameElement", "navigator",
    "origin", "external", "screen", "innerWidth", "innerHeight", "scrollX",
    "pageXOffset", "scrollY", "pageYOffset", "visualViewport", "screenX",
    "screenY", "outerWidth", "outerHeight", "devicePixelRatio",
    "clientInformation", "screenLeft", "screenTop", "styleMedia", "onsearch",
    "isSecureContext", "trustedTypes", "performance", "onappinstalled",
    "onbeforeinstallprompt", "crypto", "indexedDB", "sessionStorage",
    "localStorage", "onbeforexrselect", "onabort", "onbeforeinput",
    "onbeforematch", "onbeforetoggle", "onblur", "oncancel", "oncanplay",
    "oncanplaythrough", "onchange", "onclick", "onclose",
    "oncontentvisibilityautostatechange", "oncontextlost", "oncontextmenu",
    "oncontextrestored", "oncuechange", "ondblclick", "ondrag", "ondragend",
    "ondragenter", "ondragleave", "ondragover", "ondragstart", "ondrop",
    "ondurationchange", "onemptied", "onended", "onerror", "onfocus",
    "onformdata", "oninput", "oninvalid", "onkeydown", "onkeypress",
    "onkeyup", "onload", "onloadeddata", "onloadedmetadata", "onloadstart",
    "onmousedown", "onmouseenter", "onmouseleave", "onmousemove",
    "onmouseout", "onmouseover", "onmouseup", "onmousewheel", "onpause",
    "onplay", "onplaying", "onprogress", "onratechange", "onreset",
    "onresize", "onscroll", "onsecuritypolicyviolation", "onseeked",
    "onseeking", "onselect", "onslotchange", "onstalled", "onsubmit",
    "onsuspend", "ontimeupdate", "ontoggle", "onvolumechange", "onwaiting",
    "onwebkitanimationend", "onwebkitanimationiteration",
    "onwebkitanimationstart", "onwebkittransitionend", "onwheel",
    "onauxclick", "ongotpointercapture", "onlostpointercapture",
    "onpointerdown", "onpointermove", "onpointerrawupdate", "onpointerup",
    "onpointercancel", "onpointerover", "onpointerout", "onpointerenter",
    "onpointerleave", "onselectstart", "onselectionchange",
    "onanimationend", "onanimationiteration", "onanimationstart",
    "ontransitionrun", "ontransitionstart", "ontransitionend",
    "ontransitioncancel", "onafterprint", "onbeforeprint", "onbeforeunload",
    "onhashchange", "onlanguagechange", "onmessage", "onmessageerror",
    "onoffline", "ononline", "onpagehide", "onpageshow", "onpopstate",
    "onrejectionhandled", "onstorage", "onunhandledrejection", "onunload",
    "crossOriginIsolated", "scheduler", "alert", "atob", "blur", "btoa",
    "cancelAnimationFrame", "cancelIdleCallback", "captureEvents",
    "clearInterval", "clearTimeout", "close", "confirm", "createImageBitmap",
    "fetch", "find", "focus", "getComputedStyle", "getSelection",
    "matchMedia", "moveBy", "moveTo", "open", "postMessage", "print",
    "prompt", "queueMicrotask", "releaseEvents", "reportError",
    "requestAnimationFrame", "requestIdleCallback", "resizeBy", "resizeTo",
    "scroll", "scrollBy", "scrollTo", "setInterval", "setTimeout", "stop",
    "structuredClone", "webkitCancelAnimationFrame",
    "webkitRequestAnimationFrame", "chrome", "caches", "cookieStore",
    "ondevicemotion", "ondeviceorientation", "ondeviceorientationabsolute",
    "launchQueue", "documentPictureInPicture", "getScreenDetails",
    "queryLocalFonts", "showDirectoryPicker", "showOpenFilePicker",
    "showSaveFilePicker", "originAgentCluster", "onpageswap",
    "onpagereveal", "credentialless", "speechSynthesis", "onscrollend",
    "webkitRequestFileSystem", "webkitResolveLocalFileSystemURL",
    "sendMsgToSolverCS", "webpackChunk_N_E", "__next_set_public_path__",
    "next", "__NEXT_DATA__", "__SSG_MANIFEST_CB", "__NEXT_P", "_N_E",
    "regeneratorRuntime", "__REACT_INTL_CONTEXT__", "DD_RUM", "_",
    "filterCSS", "filterXSS", "__SEGMENT_INSPECTOR__", "__NEXT_PRELOADREADY",
    "Intercom", "__MIDDLEWARE_MATCHERS", "__STATSIG_SDK__",
    "__STATSIG_JS_SDK__", "__STATSIG_RERENDER_OVERRIDE__",
    "_oaiHandleSessionExpired", "__BUILD_MANIFEST", "__SSG_MANIFEST",
    "__intercomAssignLocation", "__intercomReloadLocation",
]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)
SENTINEL_SCRIPT_URL = "https://chatgpt.com/backend-api/sentinel/sdk.js"
MAX_ITERATIONS = 500000


def _random_choice(arr):
    return arr[random.randint(0, len(arr) - 1)]


def _build_config(user_agent: str) -> list:
    """Build the config array matching Go's buildConfig()."""
    now = time.time()
    now_ms = int(now * 1000)
    perf_counter = float(now_ms % 1000000) + random.random()
    epoch_offset = float(now_ms) - perf_counter

    # parse time: "4/29/2026, 3:45:12 PM" style
    pt = datetime.now(timezone(timedelta(hours=-7)))  # America/Los_Angeles
    parse_time = pt.strftime("%-m/%-d/%Y, %-I:%M:%S %p")  # macOS style

    return [
        _random_choice(SCREEN_SIZES),          # 0
        parse_time,                             # 1
        4294705152,                             # 2 (constant)
        0,                                      # 3 (nonce_i placeholder)
        user_agent,                             # 4
        SENTINEL_SCRIPT_URL,                    # 5
        "",                                     # 6 (dpl)
        "en-US",                                # 7
        "en-US,es-US,en,es",                    # 8
        0,                                      # 9 (nonce_j placeholder)
        _random_choice(NAVIGATOR_KEY),          # 10
        _random_choice(DOCUMENT_KEY),           # 11
        _random_choice(WINDOW_KEY),             # 12
        perf_counter,                           # 13
        str(uuid.uuid4()),                      # 14
        "",                                     # 15
        _random_choice(CORE_COUNTS),            # 16
        epoch_offset,                           # 17
    ]


def _assemble_solve(config: list, i: int, j: int) -> str:
    """Build the JSON array string, base64-encode it (matching Go solvePoW)."""
    # Build the 3 JSON fragments
    part1_json = json.dumps(config[:3])          # "[v0,v1,v2]"
    part4to8_json = json.dumps(config[4:9])      # "[v4,v5,v6,v7,v8]"
    part10_json = json.dumps(config[10:])         # "[v10,...v17]"

    static_part1 = part1_json[:-1] + ","         # "[v0,v1,v2,"
    mid = part4to8_json[1:-1]                     # "v4,v5,v6,v7,v8"
    static_part2 = "," + mid + ","
    tail = part10_json[1:]                        # "v10,...v17]"
    static_part3 = "," + tail

    assembled = f"{static_part1}{i}{static_part2}{j}{static_part3}"
    return base64.b64encode(assembled.encode()).decode()


def _bytes_le(a: bytes, b: bytes) -> bool:
    """Return True if a <= b lexicographically."""
    for x, y in zip(a, b):
        if x < y:
            return True
        if x > y:
            return False
    return len(a) <= len(b)


def solve_pow(seed: str, difficulty: str) -> str:
    """
    Solve the PoW challenge. Returns proof token (prefixed "gAAAAAB").
    Adapted from Go solvePoW().
    """
    config = _build_config(DEFAULT_USER_AGENT)
    diff_bytes = bytes.fromhex(difficulty)
    diff_len = len(diff_bytes)
    seed_bytes = seed.encode()

    for i in range(MAX_ITERATIONS):
        j = i >> 1
        b64 = _assemble_solve(config, i, j)

        h = hashlib.sha3_512(seed_bytes + b64.encode()).digest()

        if _bytes_le(h[:diff_len], diff_bytes):
            print(f"[PoW] Solved at iteration {i}")
            return "gAAAAAB" + b64

    # Fallback (matching Go)
    fallback = base64.b64encode(json.dumps(seed).encode()).decode()
    print("[PoW] Failed to solve, using fallback")
    return "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback


def generate_requirements_token() -> str:
    """
    Generate the 'p' token for /sentinel/chat-requirements request.
    Uses nonce_i=0, nonce_j=0 (one fixed iteration).
    """
    config = _build_config(DEFAULT_USER_AGENT)
    b64 = _assemble_solve(config, 0, 0)
    return "gAAAAAC" + b64
