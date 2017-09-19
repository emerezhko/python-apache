#!/usr/bin/env python
import os
import commands

table = {}
result = []
from os import stat
from pwd import getpwuid


def find_owner(filename):
    return getpwuid(stat(filename).st_uid).pw_name


apache_conf_dir = "/etc/apache2/sites-enabled/"  # Ubuntu style
listOfSite = os.listdir(apache_conf_dir)
for site in listOfSite:
    if os.path.isfile(os.path.join(apache_conf_dir) + site):
        path_to_domain = os.path.join(apache_conf_dir) + site
        with open(path_to_domain, 'r') as domain_file:
            for config in domain_file:
                if '<virtualhost *:' in config.lower():
                    if '*:80' not in config:
                        listen_ip = config.split()[-1]
                        table[site] = {"listen_ip": listen_ip[:-1]}
                if 'servername' in config.lower():
                    if site in table:
                        server_name = config.split()[-1]
                        table[site]["Server name"] = server_name
                    else:
                        server_name = config.split()[-1]
                        table[site] = {"Server name": server_name}
                if 'sslcacertificatefile' in config.lower():
                    ca_file_name = config.split()[-1]
                    table[site]["CACertificateFile"] = ca_file_name
                    owner_of_CA = find_owner(ca_file_name)
                    table[site]["CACertificateFileowner"] = owner_of_CA
                if 'serveralias' in config.lower():
                    server_alias = config.split()[1:]
                    table[site]["ServerAlias"] = server_alias
                if 'sslcertificatefile' in config.lower():
                    cert_file_name = config.split()[-1]
                    table[site]["CertificateFile"] = cert_file_name
                    owner_of_CertificateFile = find_owner(cert_file_name)
                    table[site]["CertificateFileowner"] = owner_of_CertificateFile
                if 'sslcertificatekeyfile' in config.lower():
                    key_file_name = config.split()[-1]
                    table[site]["keyfile"] = key_file_name
                    owner_of_PrivateKey = find_owner(key_file_name)
                    table[site]["PrivateKeyowner"] = owner_of_PrivateKey

for site in table:
    for key, val in table[site].items():
        if key == "CertificateFile":
            subject = commands.getoutput("openssl x509 -in  " + val + "  -noout -subject")
            table[site]["certificate_subject"] = subject.split("CN=")[-1]
            issuer = commands.getoutput("openssl x509 -in  " + val + "  -noout -issuer")
            table[site]["certificate_issuer"] = issuer.split("CN=")[-1]
            enddate = commands.getoutput("openssl x509 -in  " + val + "  -noout -enddate")
            table[site]["certificate_enddate"] = enddate.split("CN=")[-1][9:]
        if key == "CACertificateFile":
            table[site]["validate"] = commands.getoutput(
                "openssl verify -verbose -CAfile " + table[site]["CACertificateFile"] + " " + table[site][
                    "CertificateFile"])

for site in table:
    result = []
    result.append(site)
    for key, val in table[site].items():
        result.append(val)
    print str(result)[1:-1]
