import argostranslate.package
import argostranslate.translate

# update package list
argostranslate.package.update_package_index()

# get available packages
available_packages = argostranslate.package.get_available_packages()

# find English -> Hindi package
package_to_install = next(
    filter(
        lambda x: x.from_code == "en" and x.to_code == "hi",
        available_packages
    )
)

# download and install
download_path = package_to_install.download()
argostranslate.package.install_from_path(download_path)

print("English -> Hindi model installed!")