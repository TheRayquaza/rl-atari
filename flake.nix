{
  description = "Simple python fhs devshell that works";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = let
          python = pkgs.python312;
        in
          (pkgs.buildFHSEnv {
            name = "simple-python-fhs";

            targetPkgs = _: [
              python
              pkgs.uv
              pkgs.zlib

              pkgs.mesa
              pkgs.libglvnd
              pkgs.xorg.libX11
              pkgs.xorg.libXext
              pkgs.glib
              pkgs.gtk3
              pkgs.gst_all_1.gstreamer
              pkgs.gst_all_1.gst-plugins-base
              pkgs.gst_all_1.gst-plugins-good
              pkgs.qt5.qtbase

              pkgs.xorg.libxcb
              pkgs.xorg.libXrender
              pkgs.xorg.libXrandr
              pkgs.xorg.libXi
              pkgs.xorg.libXfixes
              pkgs.xorg.libXcursor
              pkgs.xorg.libXinerama
              pkgs.xorg.xcbutil
              pkgs.xorg.xcbutilimage
              pkgs.xorg.xcbutilkeysyms
              pkgs.xorg.xcbutilwm
              pkgs.xorg.xcbutilrenderutil
            ];
            profile = ''
              export UV_PYTHON=${python}
              bash
            '';
          }).env;
      }
    );
}
