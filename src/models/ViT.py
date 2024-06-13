import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, num_target_genes, pretrain=False):
        super(ViT, self).__init__()
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(768, num_target_genes)

    def forward(self, x):
        x = self.base_model(x)
        output = self.fc(x)
        return output

class ViTExtractor(nn.Module):
    def __init__(self, pretrain=False):
        super(ViTExtractor, self).__init__()
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        x = self.base_model(x)
        return x

class AuxViT(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain=False):
        super(AuxViT, self).__init__()
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(768, num_target_genes)
        self.fc2 = nn.Linear(768, num_aux_genes)

    def forward(self, x):
        x = self.base_model(x)
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2

class AnnotateViT(nn.Module):
    def __init__(self, num_target_genes, pretrain=False):
        super(AnnotateViT, self).__init__()
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(768, num_target_genes)
        self.fc2 = nn.Linear(768, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        output1 = self.fc1(x)
        output2 = torch.sigmoid(self.fc2(x))
        return output1, output2

class AuxAnnotateViT(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain=False, extraction=False):
        super(AuxAnnotateViT, self).__init__()
        self.extraction = extraction
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(768, num_target_genes)
        self.fc2 = nn.Linear(768, num_aux_genes)
        self.fc3 = nn.Linear(768, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        if self.extraction:
            return x
        else:
            output1 = self.fc1(x)
            output2 = self.fc2(x)
            output3 = torch.sigmoid(self.fc3(x))
            return output1, output2, output3

class AuxAnnotateSpatialViT(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain=False, extraction=False):
        super(AuxAnnotateSpatialViT, self).__init__()
        self.extraction = extraction
        if pretrain:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.base_model.head = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(768, num_target_genes)
        self.fc_aux = nn.Linear(768, num_aux_genes)
        self.fc_tmr = nn.Linear(768, 1)  # Binary classification head

    def forward(self, x, x_nbr):  # x_neighbor is the neighbor spots, cost more memory

        if self.training:
            bs, channels, height, width = x.shape
            num_nbr = x_nbr.shape[1]
            x_nbr = x_nbr.reshape(-1, channels, height, width)

            x_combined = torch.cat((x, x_nbr), dim=0)
            x_combined = self.base_model(x_combined)

            x = x_combined[:bs]
            x_nbr = x_combined[bs:]

            # TODO: can calculate the similarity between x and x_nbr here

            x_nbr = x_nbr.reshape(bs, num_nbr, -1)

            output = self.fc(x)  # for target genes
            output_nbr = self.fc(x_nbr)  # only predict target genes for neighbor spots, not aux genes

            target_output = self.fc_aux(x)  # for aux genes
            target_output_nbr = self.fc_aux(x_nbr)

            tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation
            tmr_output_nbr = torch.sigmoid(self.fc_tmr(x_nbr))

            return output, target_output, tmr_output, output_nbr, target_output_nbr, tmr_output_nbr

        else:
            x = self.base_model(x)
            if self.extraction:
                return x
            else:
                output = self.fc(x)  # for target genes
                target_output = self.fc_aux(x)  # for aux genes
                tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation

                return output, target_output, tmr_output

if __name__ == "__main__":
    model = AuxAnnotateSpatialViT(250, 10000, pretrain=True)
    # test the model use random input
    x = torch.randn(2, 3, 224, 224)
    x_neighbor = torch.randn(2, 8, 3, 224, 224)
    output1, output2, output3, output1_neighbor, output2_neighbor, output3_neighbor = model(x, x_neighbor)
    print(output1.shape)
