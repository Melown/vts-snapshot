/**
 * Copyright (c) 2018 Melown Technologies SE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * *  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <boost/filesystem.hpp>
#include <boost/noncopyable.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vts-browser/log.hpp>

#include "service/cmdline.hpp"
#include "utility/buildsys.hpp"
#include "utility/gccversion.hpp"
#include "utility/limits.hpp"
#include "utility/path.hpp"
#include "utility/openmp.hpp"

#include "math/geometry_core.hpp"

#include "imgproc/cvmat.hpp"

#include "geo/srsdef.hpp"
#include "geo/coordinates.hpp"

#include "vtsoffscreen/snapper.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace vo = vts::offscreen;

class Snapshot : public service::Cmdline {
public:
    Snapshot()
        : service::Cmdline("vts-snapshot", BUILD_TARGET_VERSION)
    {}

private:
    virtual void configuration(po::options_description &cmdline
                               , po::options_description &config
                               , po::positional_options_description &pd)
        override;

    virtual void configure(const po::variables_map &vars) override;

    virtual bool help(std::ostream &out, const std::string &what) const
        override;

    virtual int run() override;

    vo::SnapperConfig snapperConfig_;

    fs::path input_;
    std::string imageFileExtentsion_ = ".jpg";
    fs::path output_;
    int jpegQuality_ = 85;
};

void Snapshot::configuration(po::options_description &cmdline
                                 , po::options_description &config
                                 , po::positional_options_description &pd)
{
    config.add_options()
        ("mapConfigUrl", po::value(&snapperConfig_.mapConfigUrl)->required()
         , "URL of VTS map configuration.")
        ("authUrl", po::value(&snapperConfig_.authUrl)
         ->default_value(snapperConfig_.authUrl)->required()
         , "URL of VTS auth server configuration.")

        ("input", po::value(&input_)->required()
         , "Path to intput position file.")

        ("imageFileExtentsion", po::value(&imageFileExtentsion_)
         ->default_value(imageFileExtentsion_)->required()
         , "Output image file extentsion (i.e. image file format); "
         "one of .png, .jpg, .tif.")

        ("jpegQuality", po::value(&jpegQuality_)
         ->default_value(jpegQuality_)->required()
         , "JPEG compression quality (0-100). Applicable only "
         "when imageFileExtentsion is \".jpg\".")

        ("output", po::value(&output_)->required()
         , "Output directory.")
        ;

    pd.add("mapConfigUrl", 1);

    (void) cmdline;
}

void Snapshot::configure(const po::variables_map &vars)
{
    // fwd dbglog settings to VTS libraries
    vts::setLogMask(dbglog::get_mask_string());
    vts::setLogFile(Program::logFile().string());
    vts::setLogConsole(dbglog::get_log_console());

    (void) vars;
}

bool Snapshot::help(std::ostream &out, const std::string &what) const
{
    if (what.empty()) {
        out << R"RAW(vts-snapshot
usage
    vts-snapshot MAP-CONFIG-URL [OPTIONS]

)RAW";
    }
    return false;
}

void save(const fs::path &filename, const cv::Mat &image
          , int jpegQuality)
{
    fs::create_directories(filename.parent_path());
    cv::imwrite(filename.string(), image
                , { cv::IMWRITE_JPEG_QUALITY, jpegQuality
                ,   cv::IMWRITE_PNG_COMPRESSION, 9 });
}

int Snapshot::run()
{
    LOG(info4) << "All done.";
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    // get rid of DISPLAY variable so there is no offending X11 system in the
    // way
    ::unsetenv("DISPLAY");

    utility::unlimitedCoredump();
    return Snapshot()(argc, argv);
}